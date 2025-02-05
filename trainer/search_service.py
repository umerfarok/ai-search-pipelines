import os
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from flask import Flask, request, jsonify
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import pandas as pd
import io
from collections import OrderedDict
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
from transformers import pipeline
import json
import spacy
from config import AppConfig
import faiss
from concurrent.futures import ThreadPoolExecutor
from cachetools import TTLCache, LRUCache
import time
import pycorrector
from rapidfuzz import process, fuzz


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class ModelCache:
    """Enhanced model caching with thread safety"""

    def __init__(self, max_size: int = 4):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, model_path: str) -> Optional[Dict]:
        with self.lock:
            if model_path in self.cache:
                self.cache.move_to_end(model_path)
                return self.cache[model_path]
            return None

    def put(self, model_path: str, data: Dict):
        with self.lock:
            if model_path in self.cache:
                self.cache.move_to_end(model_path)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                self.cache[model_path] = data


class S3Manager:
    """Enhanced S3 manager with retry logic"""

    def __init__(self):
        retry_config = Config(
            retries=dict(max_attempts=AppConfig.MAX_RETRIES, mode="adaptive"),
            s3={"addressing_style": "path"},
        )

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AppConfig.AWS_ACCESS_KEY,
            aws_secret_access_key=AppConfig.AWS_SECRET_KEY,
            region_name=AppConfig.S3_REGION,
            endpoint_url=AppConfig.AWS_ENDPOINT_URL,
            verify=AppConfig.AWS_SSL_VERIFY.lower() == "true",
            config=retry_config,
        )
        self.bucket = AppConfig.S3_BUCKET

    def download_file(self, s3_path: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3.download_file(self.bucket, s3_path, local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download {s3_path}: {e}")
            return False

    def get_csv_content(self, s3_path: str) -> Optional[pd.DataFrame]:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_path)
            return pd.read_csv(io.BytesIO(response["Body"].read()))
        except Exception as e:
            logger.error(f"Failed to read CSV from S3: {e}")
            return None


class EmbeddingManager:
    """Enhanced embedding manager with model versioning"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_models = {}
        self.model_load_lock = threading.Lock()
        self.max_seq_length = 512
        self.cache_dir = AppConfig.TRANSFORMER_CACHE
        self.MODEL_MAPPINGS = AppConfig.MODEL_MAPPINGS
        self.default_model_name = "all-minilm-l6"

    def _get_model_info(self, model_name: str) -> dict:
        if model_name in self.MODEL_MAPPINGS:
            return self.MODEL_MAPPINGS[model_name]

        for key, info in self.MODEL_MAPPINGS.items():
            if info["path"] == model_name:
                return info

        for key, info in self.MODEL_MAPPINGS.items():
            if model_name.lower().replace("-", "") in info["path"].lower().replace(
                "-", ""
            ):
                return info

        raise ValueError(f"Unsupported model: {model_name}")

    def get_model(self, model_name: str = None):
        if not model_name:
            model_name = self.default_model_name

        model_info = self._get_model_info(model_name)
        model_path = model_info["path"]

        if model_path in self.embedding_models:
            return self.embedding_models[model_path]

        try:
            with self.model_load_lock:
                if model_path not in self.embedding_models:
                    model = SentenceTransformer(
                        model_path, cache_folder=self.cache_dir, device=self.device
                    )
                    self.embedding_models[model_path] = model
                return self.embedding_models[model_path]

        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise

    def generate_embedding(self, text: List[str], model_name: str = None) -> np.ndarray:
        if not text:
            raise ValueError("Empty text input")

        model = self.get_model(model_name)
        batch_size = 32

        try:
            with torch.no_grad():
                embeddings = model.encode(
                    text,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
                return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class EnhancedQueryUnderstanding:
    """Enhanced query understanding with improved preprocessing"""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device,
        )
        self.qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=self.device,
        )
        self.query_expansion_model = pipeline(
            "text2text-generation", model="google/flan-t5-base", device=self.device
        )
        self.feature_extractor = pipeline(
            "feature-extraction",
            model="sentence-transformers/all-MiniLM-L6-v2",
            device=self.device,
        )
        self.keyphrase_model = pipeline(
            "keyphrase-extraction",
            model="ml6team/keyphrase-extraction-distilbert-inspec",
            device=self.device,
        )
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        if not hasattr(self, "sentiment_analyzer"):
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device,
            )

    def _correct_spelling(self, query: str) -> str:
        """Apply spelling correction to query"""
        try:
            corrected, _ = pycorrector.correct(query)
            return corrected
        except:
            return query

    def _extract_keyphrases(self, query: str) -> List[str]:
        """Extract key phrases from query"""
        try:
            return [kp["word"] for kp in self.keyphrase_model(query)]
        except:
            return []

    def _get_dynamic_categories(self, text: str) -> List[Dict[str, float]]:
        """Dynamically extract categories from text"""
        prompts = [
            "What is the main topic or domain of this query?",
            "What specific aspects or features are being searched for?",
            "What is the user's intent or goal?",
            "What constraints or requirements are mentioned?",
        ]
        categories = []
        for prompt in prompts:
            context = f"Query: {text}\nQuestion: {prompt}"
            response = self.qa_model(question=prompt, context=context)
            if response["score"] > 0.3:
                categories.append(
                    {
                        "category": response["answer"],
                        "confidence": float(response["score"]),
                    }
                )
        return categories

    def _expand_query(self, query: str) -> List[str]:
        """Intent-aware query expansion"""
        intent_results = self.zero_shot(
            query,
            candidate_labels=["search", "compare", "explore", "purchase", "learn"],
            multi_label=True,
        )
        primary_intent = (
            intent_results["labels"][0] if intent_results["labels"] else "search"
        )

        prompt_templates = {
            "search": f"Expand this product search query with specific attributes: {query}",
            "compare": f"Generate comparison terms for: {query}",
            "explore": f"Suggest related categories for: {query}",
            "purchase": f"Generate purchase-related terms for: {query}",
            "learn": f"Expand this informational query with technical terms: {query}",
        }

        prompt = prompt_templates.get(
            primary_intent, f"Generate relevant search terms for: {query}"
        )

        try:
            expanded = self.query_expansion_model(
                prompt,
                max_length=50,
                num_return_sequences=1,
                num_beams=3,
                early_stopping=True,
            )
            expanded_terms = expanded[0]["generated_text"].split()
            return list(set([query] + expanded_terms))
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]

    def _extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features from text"""
        doc = self.nlp(text)
        return {
            "entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents],
            "key_phrases": [chunk.text for chunk in doc.noun_chunks],
            "sentiment": self._get_sentiment(doc.text),
            "dependencies": [
                {"text": token.text, "dep": token.dep_}
                for token in doc
                if token.dep_ not in ["punct", "det"]
            ],
        }

    def _get_sentiment(self, text: str) -> float:
        """Analyze text sentiment"""
        try:
            result = self.sentiment_analyzer(text[:512])[0]
            return (
                result["score"] if result["label"] == "POSITIVE" else -result["score"]
            )
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0

    def _analyze_context(self, query: str, context: Dict) -> Dict:
        """Analyze query in context"""
        context_understanding = {}

        try:
            if "previous_queries" in context:
                context_understanding["query_evolution"] = (
                    self._analyze_query_evolution(query, context["previous_queries"])
                )

            if "domain" in context:
                context_understanding["domain_relevance"] = (
                    self._analyze_domain_relevance(query, context["domain"])
                )

            if "user_preferences" in context:
                context_understanding["preference_match"] = self._analyze_preferences(
                    query, context["user_preferences"]
                )

        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            context_understanding["error"] = str(e)

        return context_understanding

    def _analyze_query_evolution(
        self, current_query: str, previous_queries: List[str]
    ) -> Dict:
        """Analyze query evolution"""
        try:
            if not previous_queries:
                return {"type": "initial_query"}

            last_query = previous_queries[-1]
            similarity = float(
                np.dot(
                    self.feature_extractor(current_query, return_tensors=True)[0].mean(
                        axis=0
                    ),
                    self.feature_extractor(last_query, return_tensors=True)[0].mean(
                        axis=0
                    ),
                )
            )

            if similarity > 0.8:
                evolution_type = "refinement"
            elif similarity > 0.5:
                evolution_type = "related"
            else:
                evolution_type = "new_topic"

            return {
                "type": evolution_type,
                "similarity": similarity,
                "previous_query": last_query,
            }

        except Exception as e:
            logger.error(f"Query evolution analysis failed: {e}")
            return {"error": str(e)}

    def analyze(self, query: str, context: Optional[Dict] = None) -> Dict:
        """Comprehensive query analysis"""
        try:
            # Enhanced preprocessing
            query = self._correct_spelling(query.lower().strip())

            # Get basic analysis
            categories = self._get_dynamic_categories(query)
            expanded_terms = self._expand_query(query)
            semantic_features = self._extract_semantic_features(query)

            # Get intent analysis
            intent_labels = ["search", "compare", "explore", "purchase", "learn"]
            intent_results = self.zero_shot(
                query, candidate_labels=intent_labels, multi_label=True
            )

            # Extract keyphrases and fuzzy matches
            keyphrases = self._extract_keyphrases(query)
            fuzzy_terms = list(
                set(
                    process.extract(
                        query, expanded_terms, limit=3, scorer=fuzz.token_set_ratio
                    )
                )
            )

            understanding = {
                "original_query": query,
                "categories": categories,
                "expanded_terms": expanded_terms,
                "semantic_features": semantic_features,
                "keyphrases": keyphrases,
                "fuzzy_terms": fuzzy_terms,
                "intent": {
                    "labels": intent_results["labels"],
                    "scores": [float(score) for score in intent_results["scores"]],
                },
            }

            if context:
                understanding["context"] = self._analyze_context(query, context)

            return understanding

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {"original_query": query, "error": str(e)}


class FastReranker:
    """Two-stage reranker with caching for optimal performance"""

    def __init__(self):
        self.light_ranker = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")
        self.heavy_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.cache = LRUCache(maxsize=1000)

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """Two-stage reranking with caching"""
        cache_key = f"{query[:50]}_{top_k}"
        if cached := self.cache.get(cache_key):
            return cached

        if not candidates:
            return []

        try:
            # First stage: light reranking
            pairs = [(query, c["text"]) for c in candidates]
            light_scores = self.light_ranker.predict(pairs, batch_size=32)
            top_indices = np.argpartition(light_scores, -50)[-50:]

            # Second stage: heavy reranking
            heavy_pairs = [(query, candidates[i]["text"]) for i in top_indices]
            heavy_scores = self.heavy_ranker.predict(heavy_pairs)

            # Combine results
            for i, score in zip(top_indices, heavy_scores):
                candidates[i]["rerank_score"] = float(score)

            reranked = sorted(
                candidates, key=lambda x: x.get("rerank_score", 0), reverse=True
            )[:top_k]
            self.cache[cache_key] = reranked
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates[:top_k]


class OptimizedHybridSearch:
    """Optimized hybrid search engine with FAISS integration"""

    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = None
        self.faiss_index = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.reranker = FastReranker()
        self.exact_match_boost = 3.0
        self.field_weights = {
            "name": 2.5,
            "category": 1.8,
            "description": 1.2,
            "specifications": 1.5,
        }

    def initialize(self, texts: List[str], embeddings: np.ndarray) -> bool:
        """Initialize search engine with corpus and FAISS index"""
        try:
            if not texts:
                logger.error("Empty text corpus")
                return False

            # Initialize BM25
            self.tokenized_corpus = []
            for text in texts:
                try:
                    tokenized = (
                        word_tokenize(text.lower()) if isinstance(text, str) else []
                    )
                    self.tokenized_corpus.append(tokenized)
                except Exception as e:
                    logger.warning(f"Error tokenizing document: {e}")
                    self.tokenized_corpus.append([])

            self.bm25 = BM25Okapi(self.tokenized_corpus)

            # Initialize FAISS
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(embeddings.astype("float32"))

            return True

        except Exception as e:
            logger.error(f"Search engine initialization failed: {e}")
            return False

    def _exact_match_boost(self, candidate: Dict, query_terms: List[str]) -> float:
        """Calculate exact match boost for a candidate"""
        boost = 1.0
        for field, weight in self.field_weights.items():
            if field in candidate:
                field_text = candidate[field].lower()
                for term in query_terms:
                    if term in field_text:
                        boost += weight * 0.2
        return min(boost, self.exact_match_boost)

    def hybrid_search(
        self, query: str, query_embed: np.ndarray, context: Dict, k: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform hybrid search with FAISS and BM25"""
        try:
            start_time = time.time()

            # First-stage FAISS search
            D, I = self.faiss_index.search(query_embed.astype("float32"), k)
            semantic_scores = D[0]

            # Get BM25 scores
            tokenized_query = word_tokenize(query.lower())
            bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
            bm25_scores = bm25_scores[I[0]]

            # Combine scores with dynamic weighting
            alpha = self._adjust_weights(context["query_analysis"])
            combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores

            # Apply exact match boosts
            query_terms = set(
                tokenized_query + context["query_analysis"].get("keyphrases", [])
            )
            boosted_scores = []

            for idx, score in zip(I[0], combined_scores):
                candidate = {
                    "text": " ".join(self.tokenized_corpus[idx]),
                    "score": score,
                }
                boost = self._exact_match_boost(candidate, query_terms)
                boosted_scores.append(score * boost)

            boosted_scores = np.array(boosted_scores)

            logger.info(f"Hybrid search completed in {time.time()-start_time:.3f}s")
            return I[0], boosted_scores

        except Exception as e:
            logger.error(f"Optimized search failed: {e}")
            return np.array([]), np.array([])

    def _adjust_weights(self, query_analysis: Dict) -> float:
        """Dynamically adjust weights based on query analysis"""
        base_alpha = 0.7
        intent_weights = {
            "learn": 0.15,
            "explore": 0.1,
            "search": -0.1,
            "compare": 0.2,
            "purchase": -0.15,
        }

        if "intent" in query_analysis:
            intent_scores = dict(
                zip(
                    query_analysis["intent"]["labels"],
                    query_analysis["intent"]["scores"],
                )
            )
            for intent, score in intent_scores.items():
                base_alpha += intent_weights.get(intent, 0) * score

        if "semantic_features" in query_analysis:
            features = query_analysis["semantic_features"]
            entity_boost = min(0.1 * len(features.get("entities", [])), 0.3)
            base_alpha += entity_boost

        return max(0.3, min(0.9, base_alpha))


class EnhancedSearchService:
    """Enhanced search service with optimizations"""

    def __init__(self):
        self.s3_manager = S3Manager()
        self.model_cache = ModelCache(AppConfig.MODEL_CACHE_SIZE)
        self.embedding_manager = EmbeddingManager()
        self.query_understanding = EnhancedQueryUnderstanding()
        self.hybrid_search = OptimizedHybridSearch()
        self.query_cache = TTLCache(maxsize=1000, ttl=300)
        self.preprocessed_embeddings = {}
        self.current_query_analysis = {}
        self._initialize_required_models()

    def _initialize_required_models(self):
        """Initialize required models and components"""
        try:
            for model_key in AppConfig.REQUIRED_MODELS:
                self.embedding_manager.get_model(model_key)
            logger.info("Successfully preloaded required models")
        except Exception as e:
            logger.error(f"Error preloading models: {e}")

    def initialize_model(self, model_path: str) -> bool:
        """Initialize model data and indices"""
        try:
            model_data = self.load_model(model_path)
            if not model_data:
                return False

            # Initialize search engine
            embeddings = model_data["embeddings"].astype("float32")
            texts = self._get_searchable_texts(model_data["products"])
            self.hybrid_search.initialize(texts, embeddings)

            # Preprocess field embeddings
            self._preprocess_field_embeddings(model_data["products"])

            return True

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False

    def _preprocess_field_embeddings(self, products: pd.DataFrame):
        """Precompute embeddings for critical fields"""
        fields = ["name", "category", "specifications"]
        for field in fields:
            if field in products.columns:
                texts = products[field].astype(str).tolist()
                embeddings = self.embedding_manager.generate_embedding(texts)
                self.preprocessed_embeddings[field] = embeddings

    def _get_searchable_texts(self, products: pd.DataFrame) -> List[str]:
        """Combine relevant fields into searchable texts"""
        texts = []
        for _, row in products.iterrows():
            text_parts = []
            for field in ["name", "category", "description", "specifications"]:
                if field in row and pd.notna(row[field]):
                    text_parts.append(str(row[field]))
            texts.append(" ".join(text_parts))
        return texts

    def load_model(self, model_path: str) -> Optional[Dict]:
        """Load model data from cache or storage"""
        try:
            cached_data = self.model_cache.get(model_path)
            if cached_data:
                return cached_data

            model_dir = os.path.join(AppConfig.BASE_MODEL_DIR, model_path)
            if os.path.exists(model_dir):
                return self._load_from_local(model_dir)
            return self._load_from_s3(model_path)

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def _load_from_local(self, model_dir: str) -> Optional[Dict]:
        """Load model data from local storage"""
        try:
            embeddings = np.load(os.path.join(model_dir, "embeddings.npy"))

            with open(os.path.join(model_dir, "metadata.json")) as f:
                metadata = json.load(f)

            products_df = pd.read_csv(os.path.join(model_dir, "products.csv"))

            return {
                "embeddings": embeddings,
                "metadata": metadata,
                "products": products_df,
                "loaded_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            return None

    def _load_from_s3(self, model_path: str) -> Optional[Dict]:
        """Load model data from S3"""
        try:
            model_dir = os.path.join(AppConfig.BASE_MODEL_DIR, model_path)
            os.makedirs(model_dir, exist_ok=True)

            required_files = ["embeddings.npy", "metadata.json", "products.csv"]
            for file in required_files:
                if not self.s3_manager.download_file(
                    f"{model_path}/{file}", os.path.join(model_dir, file)
                ):
                    return None

            return self._load_from_local(model_dir)

        except Exception as e:
            logger.error(f"Error loading from S3: {e}")
            return None

    def _process_candidate(
        self, product: pd.Series, score: float, schema: Dict
    ) -> Dict:
        """Process and enrich search candidate"""
        try:
            # Get field mappings
            id_column = schema.get("idcolumn", "id")
            field_mappings = {
                "name": schema.get("namecolumn", "name"),
                "description": schema.get("descriptioncolumn", "description"),
                "category": schema.get("categorycolumn", "category"),
            }

            # Build candidate
            candidate = {
                "id": (
                    str(product[id_column])
                    if id_column in product
                    else str(product.name)
                ),
                "score": float(score),
                "metadata": {},
                "field_matches": {},
                "intent_matches": {},
            }

            # Add mapped fields
            text_parts = []
            for field, column in field_mappings.items():
                if column in product and pd.notna(product[column]):
                    value = str(product[column])
                    candidate[field] = value
                    text_parts.append(value)

            candidate["text"] = " ".join(text_parts)

            # Add custom fields
            for field in schema.get("customcolumns", []):
                if field in product and pd.notna(product[field]):
                    candidate["metadata"][field] = str(product[field])

            # Add intent-specific scoring
            intents = self.current_query_analysis.get("intent", {}).get("labels", [])
            if "compare" in intents:
                candidate["intent_matches"]["compare"] = (
                    self._score_comparison_features(candidate)
                )
            if "purchase" in intents:
                candidate["intent_matches"]["purchase"] = self._score_purchase_features(
                    candidate
                )
            if "learn" in intents:
                candidate["intent_matches"]["learn"] = self._score_learn_features(
                    candidate
                )
            if "explore" in intents:
                candidate["intent_matches"]["explore"] = self._score_explore_features(
                    candidate
                )

            return candidate

        except Exception as e:
            logger.error(f"Error processing candidate: {e}")
            return None

    def _score_comparison_features(self, candidate: Dict) -> float:
        """Score product for comparison features"""
        comparison_attrs = ["specifications", "features", "price", "ratings"]
        return sum(1 for attr in comparison_attrs if candidate["metadata"].get(attr))

    def _score_purchase_features(self, candidate: Dict) -> float:
        """Score product for purchase readiness"""
        purchase_attrs = {
            "availability": 0.3,
            "price": 0.2,
            "shipping_options": 0.25,
            "return_policy": 0.25,
        }
        return sum(
            weight
            for attr, weight in purchase_attrs.items()
            if candidate["metadata"].get(attr)
        )

    def _score_learn_features(self, candidate: Dict) -> float:
        """Score product for technical information"""
        learn_terms = ["manual", "specification", "technical", "guide", "documentation"]
        return sum(1 for term in learn_terms if term in candidate["text"].lower())

    def _score_explore_features(self, candidate: Dict) -> float:
        """Score product for explorative features"""
        return len(
            [
                cat
                for cat in self.current_query_analysis.get("categories", [])
                if cat["category"].lower() in candidate["text"].lower()
            ]
        )

    def _find_field_matches(self, query: str, candidate: Dict) -> Dict:
        """Find exact and semantic matches in fields"""
        matches = {}
        query_terms = set(word_tokenize(query.lower()))

        for field in ["name", "category", "description"]:
            if field in candidate:
                field_text = candidate[field].lower()

                # Exact matches
                exact_matches = sum(1 for term in query_terms if term in field_text)

                # Semantic similarity if available
                semantic_score = 0.0
                if field in self.preprocessed_embeddings:
                    query_embed = self.embedding_manager.generate_embedding([query])[0]
                    scores = np.dot(self.preprocessed_embeddings[field], query_embed)
                    semantic_score = float(np.max(scores))

                matches[field] = {
                    "exact_matches": exact_matches,
                    "semantic_score": semantic_score,
                }

        return matches

    def search(
        self,
        query: str,
        model_path: str,
        context: Optional[Dict] = None,
        max_items: int = 10,
        min_score: float = 0.4,
    ) -> Dict:
        """Perform enhanced search with optimizations"""
        try:
            start_time = time.time()

            # Check cache
            cache_key = f"{model_path}_{query}_{max_items}"
            if cached := self.query_cache.get(cache_key):
                return cached

            # Load model data
            model_data = self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            # Analyze query
            query_analysis = self.query_understanding.analyze(query, context)
            self.current_query_analysis = query_analysis

            # Generate query embedding
            query_embed = self.embedding_manager.generate_embedding([query])[0]

            # Perform hybrid search
            indices, scores = self.hybrid_search.hybrid_search(
                query=query,
                query_embed=query_embed.reshape(1, -1),
                context={"query_analysis": query_analysis},
                k=max_items * 2,
            )

            # Process candidates
            candidates = []
            for idx, score in zip(indices, scores):
                if score >= min_score:
                    candidate = self._process_candidate(
                        model_data["products"].iloc[idx],
                        score,
                        model_data["metadata"]["schema_mapping"],
                    )
                    if candidate:
                        candidate["field_matches"] = self._find_field_matches(
                            query, candidate
                        )
                        candidates.append(candidate)

            # Rerank results
            reranked_results = self.hybrid_search.reranker.rerank(
                query, candidates, max_items
            )

            # Prepare response
            results = {
                "results": reranked_results,
                "total": len(reranked_results),
                "query_info": {
                    "original": query,
                    "analysis": query_analysis,
                    "timing": {"total_time": time.time() - start_time},
                    "model": {
                        "name": model_data["metadata"]["models"]["embeddings"],
                        "path": model_path,
                    },
                },
            }

            # Cache results
            self.query_cache[cache_key] = results

            return results

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {"error": str(e)}


# Initialize service
search_service = EnhancedSearchService()


@app.route("/search", methods=["POST"])
def search():
    """Enhanced search endpoint with improved error handling"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request data"}), 400

        # Validate required fields
        required_fields = ["query", "model_path"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return (
                jsonify(
                    {"error": f"Missing required fields: {', '.join(missing_fields)}"}
                ),
                400,
            )

        # Perform search
        try:
            results = search_service.search(
                query=data["query"],
                model_path=data["model_path"],
                context=data.get("context"),
                max_items=data.get("max_items", 10),
                min_score=data.get("min_score", 0.4),
            )
            return jsonify(results)

        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            logger.error(f"Search processing error: {str(e)}")
            return jsonify({"error": "Search processing failed"}), 500

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in request"}), 400
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/health")
def health():
    """Enhanced health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "models_loaded": list(
                search_service.embedding_manager.embedding_models.keys()
            ),
            "cache_stats": {
                "model_cache_size": len(search_service.model_cache.cache),
                "query_cache_size": len(search_service.query_cache),
            },
            "preprocessed_fields": list(search_service.preprocessed_embeddings.keys()),
            "embedding_device": str(search_service.embedding_manager.device),
        }
    )


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """Clear all caches endpoint"""
    try:
        search_service.query_cache.clear()
        search_service.model_cache.cache.clear()
        search_service.hybrid_search.reranker.cache.clear()
        return jsonify({"status": "success", "message": "All caches cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def initialize_service():
    """Initialize enhanced search service"""
    logger.info("Initializing enhanced search service...")

    try:
        # Initialize NLTK data
        nltk_data_dir = os.getenv("NLTK_DATA", "/app/cache/nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)

        try:
            nltk.download("punkt", download_dir=nltk_data_dir)
            nltk.download("punkt_tab", download_dir=nltk_data_dir)
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {e}")
            return False

        # Initialize search service
        search_service._initialize_required_models()

        logger.info("Enhanced search service initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False


if __name__ == "__main__":
    if not initialize_service():
        logger.error("Failed to initialize service")
        exit(1)

    app.run(host="0.0.0.0", port=AppConfig.SERVICE_PORT, debug=False, threaded=True)

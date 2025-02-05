"""
Enhanced Search Service with Dynamic Query Understanding and Context Extraction
"""

# Tested and working on local machine and vM
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
from spelling_corrector import SpellingCorrector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class DynamicQueryUnderstanding:
    """Enhanced query understanding with dynamic context extraction"""

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
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize spelling corrector
        self.spelling_corrector = SpellingCorrector()

    def initialize_spelling_corrector(self, custom_vocabulary: List[str] = None):
        """Initialize spelling corrector with domain-specific vocabulary"""
        self.spelling_corrector.initialize(custom_vocabulary)

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
        """Intent-aware query expansion without recursive analyze call"""
        # Get intent directly without full analysis
        intent_results = self.zero_shot(
            query,
            candidate_labels=["search", "compare", "explore", "purchase", "learn"],
            multi_label=True,
        )
        primary_intent = (
            intent_results["labels"][0] if intent_results["labels"] else "search"
        )

        prompt = {
            "search": f"Expand this product search query with specific attributes: {query}",
            "compare": f"Generate comparison terms for: {query}",
            "explore": f"Suggest related categories for: {query}",
            "purchase": f"Generate purchase-related terms for: {query}",
            "learn": f"Expand this informational query with technical terms: {query}",
        }.get(primary_intent, f"Generate relevant search terms for: {query}")

        try:
            expanded = self.query_expansion_model(
                prompt,
                max_length=50,
                num_return_sequences=1,
                num_beams=3,
                early_stopping=True,
            )
            return list(set([query] + expanded[0]["generated_text"].split()))
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]

    def _get_sentiment(self, text: str) -> float:
        """Analyze text sentiment with explicit model"""
        if not hasattr(self, "sentiment_analyzer"):
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device,
            )
        try:
            result = self.sentiment_analyzer(text[:512])[0]  # Truncate to max length
            return (
                result["score"] if result["label"] == "POSITIVE" else -result["score"]
            )
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0

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

    def analyze(self, query: str, context: Optional[Dict] = None) -> Dict:
        """Comprehensive query analysis with spelling correction"""
        try:
            # Perform spelling correction
            corrected_query, spelling_info = self.spelling_corrector.correct_spelling(
                query
            )

            # Get categories using corrected query
            categories = self._get_dynamic_categories(corrected_query)
            expanded_terms = self._expand_query(corrected_query)
            semantic_features = self._extract_semantic_features(corrected_query)

            # Get intent using corrected query
            intent_labels = ["search", "compare", "explore", "purchase", "learn"]
            intent_results = self.zero_shot(
                corrected_query, candidate_labels=intent_labels, multi_label=True
            )

            understanding = {
                "original_query": query,
                "spelling_correction": spelling_info,
                "corrected_query": corrected_query,
                "categories": categories,
                "expanded_terms": expanded_terms,
                "semantic_features": semantic_features,
                "intent": {
                    "labels": intent_results["labels"],
                    "scores": [float(score) for score in intent_results["scores"]],
                },
            }

            if context:
                understanding["context"] = self._analyze_context(
                    corrected_query, context
                )

            return understanding

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {"original_query": query, "error": str(e)}

    def _get_sentiment(self, text: str) -> float:
        """Analyze text sentiment using transformers pipeline"""
        sentiment_analyzer = pipeline("sentiment-analysis", device=self.device)
        return sentiment_analyzer(text)[0]["score"]

    def _analyze_context(self, query: str, context: Dict) -> Dict:
        """Analyze query in relation to provided context"""
        context_understanding = {}

        try:
            # Analyze query-context relevance
            if "previous_queries" in context:
                context_understanding["query_evolution"] = (
                    self._analyze_query_evolution(query, context["previous_queries"])
                )

            # Analyze domain context if available
            if "domain" in context:
                context_understanding["domain_relevance"] = (
                    self._analyze_domain_relevance(query, context["domain"])
                )

            # Add user preferences if available
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
        """Analyze how the query has evolved from previous queries"""
        try:
            if not previous_queries:
                return {"type": "initial_query"}

            # Compare with most recent query
            last_query = previous_queries[-1]

            # Get semantic similarity
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

            # Determine evolution type
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

    def _analyze_domain_relevance(self, query: str, domain: str) -> Dict:
        """Analyze query relevance to specific domain"""
        try:
            # Get domain relevance score
            relevance = self.zero_shot(
                query, candidate_labels=[domain], multi_label=False
            )

            return {
                "domain": domain,
                "relevance_score": float(relevance["scores"][0]),
                "is_relevant": relevance["scores"][0] > 0.5,
            }

        except Exception as e:
            logger.error(f"Domain relevance analysis failed: {e}")
            return {"error": str(e)}

    def _analyze_preferences(self, query: str, preferences: Dict[str, Any]) -> Dict:
        """Analyze how query matches user preferences"""
        try:
            matches = {}
            for pref_type, pref_value in preferences.items():
                # Check if preference appears in query
                relevance = self.zero_shot(
                    query, candidate_labels=[str(pref_value)], multi_label=False
                )
                matches[pref_type] = {
                    "value": pref_value,
                    "match_score": float(relevance["scores"][0]),
                }

            return matches

        except Exception as e:
            logger.error(f"Preference analysis failed: {e}")
            return {"error": str(e)}


class HybridSearchEngine:
    """Enhanced hybrid search engine with intent-aware scoring"""

    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = None
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.query_understanding = DynamicQueryUnderstanding()

    def _get_ranking_factors(self, candidate: Dict, query_analysis: Dict) -> Dict:
        """Get factors influencing the ranking"""
        return {
            "category_matches": [
                c
                for c in query_analysis.get("categories", [])
                if c["category"].lower() in candidate["text"].lower()
            ],
            "entity_matches": [
                e
                for e in query_analysis.get("semantic_features", {}).get("entities", [])
                if e["text"].lower() in candidate["text"].lower()
            ],
        }

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Safe normalization with epsilon"""
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        if np.ptp(scores) < 1e-6:  # Prevent division by zero
            return np.zeros_like(scores)
        return (scores - scores.min()) / np.ptp(scores)

    def initialize(self, texts: List[str]) -> bool:
        """Initialize search engine with corpus"""
        try:
            if not texts:
                logger.error("Empty text corpus")
                return False

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
            return True

        except Exception as e:
            logger.error(f"Search engine initialization failed: {e}")
            return False

    def hybrid_search(
        self,
        query: str,
        semantic_scores: np.ndarray,
        context: Optional[Dict] = None,
        alpha: float = 0.7,
    ) -> Tuple[np.ndarray, Dict]:
        """Enhanced hybrid search with intent-aware scoring"""
        try:
            if self.bm25 is None:
                logger.warning("BM25 not initialized. Call initialize() first.")
            query_analysis = self.query_understanding.analyze(query, context)
            tokenized_query = word_tokenize(query.lower())
            bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

            bm25_scores = self._normalize_scores(bm25_scores)
            semantic_scores = self._normalize_scores(semantic_scores)

            alpha = self._adjust_weights(query_analysis)
            combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores

            return combined_scores, {
                "query_understanding": query_analysis,
                "scores": {
                    "semantic": semantic_scores.mean(),
                    "lexical": bm25_scores.mean(),
                    "combined": combined_scores.mean(),
                },
                "weights": {"semantic": alpha, "lexical": 1 - alpha},
            }

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return semantic_scores, {"error": str(e)}

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
        if query_analysis.get("intent", {}).get("labels"):
            primary_intent = query_analysis["intent"]["labels"][0]
        if primary_intent in ["learn", "compare"]:
            base_alpha = max(base_alpha, 0.6)
        return max(0.3, min(0.9, base_alpha))

    def rerank(
        self, query: str, candidates: List[Dict], query_analysis: Dict, top_k: int = 10
    ) -> List[Dict]:
        """Intent-aware reranking"""
        if not candidates:
            return []

        try:
            pairs = [(query, doc["text"]) for doc in candidates]
            scores = self.reranker.predict(pairs)
            adjusted_scores = self._adjust_scores(scores, candidates, query_analysis)

            for i, score in enumerate(adjusted_scores):
                candidates[i]["rerank_score"] = float(score)
                candidates[i]["ranking_factors"] = self._get_ranking_factors(
                    candidates[i], query_analysis
                )

            return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[
                :top_k
            ]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates[:top_k]

    def _adjust_scores(
        self, base_scores: np.ndarray, candidates: List[Dict], query_analysis: Dict
    ) -> np.ndarray:
        """Adjust scores based on query understanding"""
        scores = np.array(base_scores)
        intent_scores = (
            dict(
                zip(
                    query_analysis["intent"]["labels"],
                    query_analysis["intent"]["scores"],
                )
            )
            if "intent" in query_analysis
            else {}
        )

        for i, candidate in enumerate(candidates):
            boost = 1.0

            # Intent-specific boosting
            if intent_scores.get("compare", 0) > 0.3:
                compare_attrs = ["specifications", "features", "price"]
                boost += 0.1 * sum(
                    1 for attr in compare_attrs if attr in candidate.get("metadata", {})
                )

            if intent_scores.get("purchase", 0) > 0.3:
                purchase_attrs = ["in_stock", "price", "delivery_options"]
                boost += 0.15 * sum(
                    1
                    for attr in purchase_attrs
                    if candidate.get("metadata", {}).get(attr)
                )

            # Existing boosts
            for category in query_analysis.get("categories", []):
                if category["category"].lower() in candidate["text"].lower():
                    boost += category["confidence"] * 0.2

            for entity in query_analysis.get("semantic_features", {}).get(
                "entities", []
            ):
                if entity["text"].lower() in candidate["text"].lower():
                    boost += 0.1

            scores[i] *= boost

        return (scores - scores.min()) / (scores.max() - scores.min())


class S3Manager:
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


class ModelCache:
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


class EmbeddingManager:
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


class SearchService:
    """Enhanced search service with intent-aware processing"""

    def __init__(self):
        self.s3_manager = S3Manager()
        self.model_cache = ModelCache(AppConfig.MODEL_CACHE_SIZE)
        self.embedding_manager = EmbeddingManager()
        self.hybrid_search = HybridSearchEngine()
        self.current_query_analysis = {}
        self._initialize_required_models()

    def _load_domain_vocabulary(self) -> List[str]:
        """Load domain-specific vocabulary for spelling correction"""
        try:
            vocab_set = set()

            # Load from products data if available
            products_df = self._load_products(AppConfig.DEFAULT_MODEL_PATH)
            if products_df is not None:
                # Extract words from relevant columns
                text_columns = ["name", "description", "category"]
                for col in text_columns:
                    if col in products_df.columns:
                        words = " ".join(products_df[col].astype(str)).lower().split()
                        vocab_set.update(words)

            # Add any additional domain-specific terms
            if hasattr(AppConfig, "DOMAIN_VOCABULARY"):
                vocab_set.update(AppConfig.DOMAIN_VOCABULARY)

            return list(vocab_set)
        except Exception as e:
            logger.error(f"Error loading domain vocabulary: {e}")
            return []

    def _initialize_required_models(self):
        """Preload required models and initialize spelling correction"""
        try:
            # Preload models from config
            for model_key in AppConfig.REQUIRED_MODELS:
                self.embedding_manager.get_model(model_key)

            # Initialize spelling correction with domain vocabulary
            custom_vocab = self._load_domain_vocabulary()
            self.hybrid_search.query_understanding.initialize_spelling_corrector(
                custom_vocab
            )

            logger.info(
                "Successfully preloaded required models and initialized spelling correction"
            )
        except Exception as e:
            logger.error(
                f"Error preloading models and initializing spelling correction: {e}"
            )

    def _load_products(self, model_path: str) -> Optional[pd.DataFrame]:
        """Load products data from local or S3"""
        local_path = os.path.join(AppConfig.BASE_MODEL_DIR, model_path, "products.csv")
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        return self.s3_manager.get_csv_content(f"{model_path}/products.csv")

    def load_model(self, model_path: str) -> Optional[Dict]:
        """Load model data from cache or storage"""
        try:
            cached_data = self.model_cache.get(model_path)
            if cached_data:
                return cached_data

            model_dir = os.path.join(AppConfig.BASE_MODEL_DIR, model_path)
            if os.path.exists(model_dir):
                return self._load_from_local(model_dir, model_path)

            return self._load_from_s3(model_path)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def _load_from_local(self, model_dir: str, model_path: str) -> Optional[Dict]:
        """Load model data from local storage"""
        try:
            embeddings = np.load(os.path.join(model_dir, "embeddings.npy"))
            with open(os.path.join(model_dir, "metadata.json")) as f:
                metadata = json.load(f)
            return {
                "embeddings": embeddings,
                "metadata": metadata,
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
            return self._load_from_local(model_dir, model_path)
        except Exception as e:
            logger.error(f"Error loading from S3: {e}")
            return None

    def _process_candidate(
        self, product: pd.Series, schema: Dict, score: float, idx: int
    ) -> Dict:
        """Process candidate with intent-aware scoring"""
        try:
            # Base processing
            text_parts = []
            id_column = schema.get("idcolumn", "ratings")
            candidate_id = (
                str(product[id_column]) if id_column in product else str(product.name)
            )

            field_mappings = {
                "name": schema.get("namecolumn", "name"),
                "description": schema.get("descriptioncolumn", "sub_category"),
                "category": schema.get("categorycolumn", "main_category"),
            }

            candidate = {
                "id": candidate_id,
                "score": score,
                "metadata": {},
                "intent_matches": {},
            }

            # Add fields and text
            for field, column in field_mappings.items():
                candidate[field] = str(product[column]) if column in product else ""
                if candidate[field]:
                    text_parts.append(candidate[field])
            candidate["text"] = " ".join(text_parts)

            # Add custom fields
            for field in schema.get("customcolumns", []):
                if field in product:
                    candidate["metadata"][field] = str(product[field])

            # Intent-specific scoring
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
            logger.error(f"Error processing candidate: {str(e)}")
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
        learn_terms = ["manual", "specification", "technical"]
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

    def search(
        self,
        query: str,
        model_path: str,
        context: Optional[Dict] = None,
        max_items: int = 10,
        min_score: float = 0.4,
    ) -> Dict:
        """Enhanced intent-aware search"""
        try:
            # Load model and data
            model_data = self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            products_df = self._load_products(model_path)
            if products_df is None:
                raise ValueError("Failed to load products data")

            # Generate embeddings and scores
            model_name = model_data["metadata"]["models"]["embeddings"]
            query_embed = self.embedding_manager.generate_embedding([query], model_name)
            semantic_scores = np.dot(model_data["embeddings"], query_embed.reshape(-1))

            # Initialize search context
            context = context or {}
            context.setdefault("previous_queries", []).append(query)

            # Perform hybrid search
            combined_scores, query_analysis = self.hybrid_search.hybrid_search(
                query, semantic_scores, context=context
            )
            self.current_query_analysis = query_analysis

            # Get and process candidates
            top_k_idx = np.argsort(combined_scores)[-max_items * 2 :][::-1]
            candidates = []
            for idx in top_k_idx:
                score = float(combined_scores[idx])
                if score >= min_score:
                    candidate = self._process_candidate(
                        products_df.iloc[idx],
                        model_data["metadata"]["schema_mapping"],
                        score,
                        idx,
                    )
                    if candidate:
                        candidates.append(candidate)

            # Rerank results
            reranked = self.hybrid_search.rerank(
                query, candidates, query_analysis, max_items
            )

            return {
                "results": reranked[:max_items],
                "total": len(reranked),
                "query_info": {
                    "original": query,
                    "analysis": query_analysis,
                    "context": context,
                    "model": {"name": model_name, "path": model_path},
                },
            }

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {"error": str(e)}


# Initialize service

search_service = SearchService()


@app.route("/search", methods=["POST"])
def search():
    """Enhanced search endpoint with improved error handling"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing request data"}), 400

        if "query" not in data:
            return jsonify({"error": "Missing required field: query"}), 400

        if "model_path" not in data:
            return jsonify({"error": "Missing required field: model_path"}), 400

        try:
            results = search_service.search(
                query=data["query"],
                model_path=data["model_path"],
                context=data.get("context"),
                max_items=data.get("max_items", 10),
                min_score=data.get("min_score", 0.4),
            )
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as e:
            logger.error(f"Search processing error: {str(e)}")
            return jsonify({"error": "Search processing failed"}), 500

        return jsonify(results)

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in request"}), 400
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "models_loaded": list(
                search_service.embedding_manager.embedding_models.keys()
            ),
        }
    )


def initialize_service():
    """Initialize search service"""
    logger.info("Initializing search service...")

    try:

        nltk_data_dir = os.getenv("NLTK_DATA", "/app/cache/nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)

        # Now download required NLTK data
        try:
            nltk.download("punkt", download_dir=nltk_data_dir)
            nltk.download("punkt_tab", download_dir=nltk_data_dir)
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {e}")
            return False

        # Initialize search service
        search_service._initialize_required_models()

        logger.info("Search service initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False


if __name__ == "__main__":
    if not initialize_service():
        logger.error("Failed to initialize service")
        exit(1)

    app.run(host="0.0.0.0", port=AppConfig.SERVICE_PORT, debug=False)

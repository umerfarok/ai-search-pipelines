"""
Enhanced Search Service with Dynamic Query Understanding and Context Extraction
"""

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
from transformers import (
    pipeline
)
import json
import spacy
from config import AppConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class DynamicQueryUnderstanding:
    """Enhanced query understanding with dynamic context extraction"""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1

        # Zero-shot classifier for flexible classification
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device,
        )

        # Question answering model for context extraction
        self.qa_model = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=self.device,
        )

        # Text generation model for query expansion
        self.query_expansion_model = pipeline(
            "text2text-generation", model="google/flan-t5-small", device=self.device
        )

        # Feature extraction for semantic understanding
        self.feature_extractor = pipeline(
            "feature-extraction",
            model="sentence-transformers/all-MiniLM-L6-v2",
            device=self.device,
        )

        # Load spaCy for linguistic analysis
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

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
        """Dynamically expand query with relevant terms"""
        prompt = f"Generate relevant search terms for: {query}"

        try:
            expanded = self.query_expansion_model(
                prompt, max_length=50, num_return_sequences=3
            )

            terms = []
            for result in expanded:
                terms.extend(result["generated_text"].split())
            return list(set(terms))

        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return []

    def _extract_semantic_features(self, text: str) -> Dict[str, Any]:
        """Extract semantic features from text"""
        doc = self.nlp(text)

        features = {
            "entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents],
            "key_phrases": [chunk.text for chunk in doc.noun_chunks],
            "sentiment": doc.sentiment,
            "dependencies": [
                {"text": token.text, "dep": token.dep_}
                for token in doc
                if token.dep_ not in ["punct", "det"]
            ],
        }

        return features

    def analyze(self, query: str, context: Optional[Dict] = None) -> Dict:
        """Comprehensive query analysis with dynamic understanding"""
        try:
            # Get dynamic categories
            categories = self._get_dynamic_categories(query)

            # Expand query terms
            expanded_terms = self._expand_query(query)

            # Extract semantic features
            semantic_features = self._extract_semantic_features(query)

            # Perform zero-shot classification for intent
            intent_labels = ["search", "compare", "explore", "purchase", "learn"]
            intent_results = self.zero_shot(
                query, candidate_labels=intent_labels, multi_label=True
            )

            # Build query understanding
            understanding = {
                "original_query": query,
                "categories": categories,
                "expanded_terms": expanded_terms,
                "semantic_features": semantic_features,
                "intent": {
                    "labels": intent_results["labels"],
                    "scores": [float(score) for score in intent_results["scores"]],
                },
                "query_vector": self.feature_extractor(query, return_tensors=True)[0]
                .mean(axis=0)
                .tolist(),
            }

            # Add context if provided
            if context:
                understanding["context"] = self._analyze_context(query, context)

            return understanding

        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {"original_query": query, "error": str(e)}

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
    """Enhanced hybrid search engine with dynamic scoring"""

    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = None
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.query_understanding = DynamicQueryUnderstanding()

    def initialize(self, texts: List[str]):
        """Initialize search engine with corpus"""
        try:
            if not texts:
                raise ValueError("Empty text corpus")

            self.tokenized_corpus = [word_tokenize(text.lower()) for text in texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            raise

    def hybrid_search(
        self,
        query: str,
        semantic_scores: np.ndarray,
        context: Optional[Dict] = None,
        alpha: float = 0.7,
    ) -> Tuple[np.ndarray, Dict]:
        """Enhanced hybrid search with contextual understanding"""
        if not query.strip():
            return np.zeros_like(semantic_scores), {}

        try:
            # Get query understanding
            query_analysis = self.query_understanding.analyze(query, context)

            # Get BM25 scores
            tokenized_query = word_tokenize(query.lower())
            bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

            # Normalize scores
            bm25_scores = self._normalize_scores(bm25_scores)
            semantic_scores = self._normalize_scores(semantic_scores)

            # Adjust weights based on query understanding
            alpha = self._adjust_weights(query_analysis)

            # Combine scores
            combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores

            return combined_scores, query_analysis

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return semantic_scores, {"error": str(e)}

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize score array"""
        score_range = scores.max() - scores.min()
        if score_range > 0:
            return (scores - scores.min()) / score_range
        return scores

    def _adjust_weights(self, query_analysis: Dict) -> float:
        """Dynamically adjust weights based on query analysis"""
        try:
            base_alpha = 0.7

            # Adjust based on intent
            if "intent" in query_analysis:
                intent_scores = dict(
                    zip(
                        query_analysis["intent"]["labels"],
                        query_analysis["intent"]["scores"],
                    )
                )

                # Increase semantic weight for conceptual queries
                if (
                    intent_scores.get("learn", 0) > 0.5
                    or intent_scores.get("explore", 0) > 0.5
                ):
                    base_alpha += 0.1

                # Increase lexical weight for specific searches
                if intent_scores.get("search", 0) > 0.7:
                    base_alpha -= 0.1

            # Adjust based on query complexity
            if "semantic_features" in query_analysis:
                features = query_analysis["semantic_features"]

                # More entities suggest more semantic understanding needed
                if len(features.get("entities", [])) > 2:
                    base_alpha += 0.05

                # More dependencies suggest more complex relationships
                if len(features.get("dependencies", [])) > 5:
                    base_alpha += 0.05

            return max(0.3, min(0.9, base_alpha))

        except Exception as e:
            logger.error(f"Weight adjustment failed: {e}")
            return 0.7

    def rerank(
        self, query: str, candidates: List[Dict], query_analysis: Dict, top_k: int = 10
    ) -> List[Dict]:
        """Enhanced reranking with query understanding"""
        if not candidates:
            return []

        try:
            # Get base reranking scores
            pairs = [(query, doc["text"]) for doc in candidates]
            scores = self.reranker.predict(pairs)

            # Adjust scores based on query understanding
            adjusted_scores = self._adjust_scores(scores, candidates, query_analysis)

            # Apply scores and sort
            for i, score in enumerate(adjusted_scores):
                candidates[i]["rerank_score"] = float(score)
                candidates[i]["ranking_factors"] = self._get_ranking_factors(
                    candidates[i], query_analysis
                )

            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return candidates[:top_k]

    def _adjust_scores(
        self, base_scores: np.ndarray, candidates: List[Dict], query_analysis: Dict
    ) -> np.ndarray:
        """Adjust scores based on query understanding"""
        try:
            scores = np.array(base_scores)

            for i, candidate in enumerate(candidates):
                boost = 1.0

                # Boost based on category match
                if query_analysis.get("categories"):
                    for category in query_analysis["categories"]:
                        if category["category"].lower() in candidate["text"].lower():
                            boost += category["confidence"] * 0.2

                # Boost based on semantic features
                if "semantic_features" in query_analysis:
                    # Entity matching
                    for entity in query_analysis["semantic_features"].get(
                        "entities", []
                    ):
                        if entity["text"].lower() in candidate["text"].lower():
                            boost += 0.1

                    # Key phrase matching
                    for phrase in query_analysis["semantic_features"].get(
                        "key_phrases", []
                    ):
                        if phrase.lower() in candidate["text"].lower():
                            boost += 0.05

                # Boost based on expanded terms
                for term in query_analysis.get("expanded_terms", []):
                    if term.lower() in candidate["text"].lower():
                        boost += 0.05

                # Apply context-based boosting if available
                if "context" in query_analysis:
                    boost *= self._apply_context_boost(
                        candidate, query_analysis["context"]
                    )

                scores[i] *= boost

            # Normalize adjusted scores
            score_range = scores.max() - scores.min()
            if score_range > 0:
                scores = (scores - scores.min()) / score_range

            return scores

        except Exception as e:
            logger.error(f"Score adjustment failed: {e}")
            return base_scores

    def _apply_context_boost(self, candidate: Dict, context: Dict) -> float:
        """Apply context-based score boosting"""
        try:
            boost = 1.0

            # Boost based on query evolution
            if "query_evolution" in context:
                evolution = context["query_evolution"]
                if evolution["type"] == "refinement":
                    # For query refinements, boost results similar to previous successful results
                    boost *= 1.2
                elif evolution["type"] == "new_topic":
                    # For new topics, reduce boost for results similar to previous queries
                    boost *= 0.9

            # Boost based on domain relevance
            if "domain_relevance" in context:
                domain_info = context["domain_relevance"]
                if domain_info.get("is_relevant", False):
                    boost *= 1 + domain_info.get("relevance_score", 0) * 0.3

            # Boost based on user preferences
            if "preference_match" in context:
                for pref_match in context["preference_match"].values():
                    if pref_match.get("match_score", 0) > 0.7:
                        boost *= 1.1

            return boost

        except Exception as e:
            logger.error(f"Context boost calculation failed: {e}")
            return 1.0

    def _get_ranking_factors(self, candidate: Dict, query_analysis: Dict) -> Dict:
        """Extract factors that influenced the ranking"""
        try:
            factors = {
                "category_matches": [],
                "entity_matches": [],
                "key_phrase_matches": [],
                "expanded_term_matches": [],
                "context_influences": [],
            }

            # Category matches
            for category in query_analysis.get("categories", []):
                if category["category"].lower() in candidate["text"].lower():
                    factors["category_matches"].append(
                        {
                            "category": category["category"],
                            "confidence": category["confidence"],
                        }
                    )

            # Entity matches
            if "semantic_features" in query_analysis:
                for entity in query_analysis["semantic_features"].get("entities", []):
                    if entity["text"].lower() in candidate["text"].lower():
                        factors["entity_matches"].append(entity)

            # Phrase matches
            for phrase in query_analysis["semantic_features"].get("key_phrases", []):
                if phrase.lower() in candidate["text"].lower():
                    factors["key_phrase_matches"].append(phrase)

            # Expanded term matches
            for term in query_analysis.get("expanded_terms", []):
                if term.lower() in candidate["text"].lower():
                    factors["expanded_term_matches"].append(term)

            # Context influences
            if "context" in query_analysis:
                context = query_analysis["context"]

                if "query_evolution" in context:
                    factors["context_influences"].append(
                        {"type": "query_evolution", "info": context["query_evolution"]}
                    )

                if "domain_relevance" in context:
                    factors["context_influences"].append(
                        {
                            "type": "domain_relevance",
                            "info": context["domain_relevance"],
                        }
                    )

            return factors

        except Exception as e:
            logger.error(f"Ranking factor extraction failed: {e}")
            return {}


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
    """Enhanced search service with dynamic query understanding"""

    def load_model(self, model_path: str) -> Optional[Dict]:
        """Load model data with enhanced error handling"""
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
            raise

    def _load_from_local(self, model_dir: str, model_path: str) -> Optional[Dict]:
        """Load model data from local storage"""
        try:
            # Load embeddings
            embeddings_path = os.path.join(model_dir, "embeddings.npy")
            embeddings = np.load(embeddings_path)

            # Load and validate metadata
            with open(os.path.join(model_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)

            if "models" not in metadata or "embeddings" not in metadata["models"]:
                raise ValueError("Invalid metadata: missing model information")

            data = {
                "embeddings": embeddings,
                "metadata": metadata,
                "loaded_at": datetime.now().isoformat(),
            }

            self.model_cache.put(model_path, data)
            return data

        except Exception as e:
            logger.error(f"Error loading from local: {e}")
            return None

    def _load_from_s3(self, model_path: str) -> Optional[Dict]:
        """Load model data from S3"""
        try:
            model_dir = os.path.join(AppConfig.BASE_MODEL_DIR, model_path)
            os.makedirs(model_dir, exist_ok=True)

            required_files = [
                "embeddings.npy",
                "metadata.json",
                "products.csv",
                "processed_texts.json",
            ]

            for file in required_files:
                s3_path = f"{model_path}/{file}"
                local_path = os.path.join(model_dir, file)

                if not self.s3_manager.download_file(s3_path, local_path):
                    logger.error(f"Failed to download {file}")
                    return None

            return self._load_from_local(model_dir, model_path)

        except Exception as e:
            logger.error(f"Error loading from S3: {e}")
            return None

    def __init__(self):
        self.s3_manager = S3Manager()
        self.model_cache = ModelCache(AppConfig.MODEL_CACHE_SIZE)
        self.embedding_manager = EmbeddingManager()
        self.hybrid_search = HybridSearchEngine()
        self._initialize_required_models()

    def search(
        self,
        query: str,
        model_path: str,
        context: Optional[Dict] = None,
        max_items: int = 10,
        min_score: float = 0.0,
    ) -> Dict:
        """Enhanced search with dynamic query understanding and contextual ranking"""
        try:
            if not query.strip():
                return {
                    "results": [],
                    "total": 0,
                    "query_info": {
                        "original": query,
                        "error": "Empty query",
                        "model_path": model_path,
                    },
                }

            # Load model data
            model_data = self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            # Generate query embedding
            model_name = model_data["metadata"]["models"]["embeddings"]
            query_embed = self.embedding_manager.generate_embedding([query], model_name)
            query_embedding = query_embed.reshape(-1)

            # Initialize hybrid search if needed
            if not self.hybrid_search.bm25:
                texts_path = os.path.join(
                    AppConfig.BASE_MODEL_DIR, model_path, "processed_texts.json"
                )
                with open(texts_path, "r") as f:
                    processed_texts = json.load(f)
                self.hybrid_search.initialize(processed_texts)

            # Calculate similarities with context
            semantic_scores = np.dot(model_data["embeddings"], query_embedding)
            combined_scores, query_analysis = self.hybrid_search.hybrid_search(
                query, semantic_scores, context=context
            )

            # Get top candidates
            top_k_idx = np.argsort(combined_scores)[-max_items * 2 :][::-1]

            # Load products
            products_df = self._load_products(model_path)
            if products_df is None:
                raise ValueError("Failed to load products data")

            # Prepare candidates for reranking
            candidates = []
            schema = model_data["metadata"]["schema_mapping"]

            for idx in top_k_idx:
                try:
                    score = float(combined_scores[idx])
                    if score < min_score:
                        continue

                    product = products_df.iloc[idx]
                    text_parts = []

                    # Get text fields based on schema
                    for field in ["name", "description", "category"]:
                        col_name = schema.get(f"{field}_column", field)
                        if col_name in product:
                            text_parts.append(str(product[col_name]))

                    candidate = {
                        "id": str(product[schema.get("id_column", "id")]),
                        "name": str(product[schema.get("name_column", "name")]),
                        "description": str(
                            product[schema.get("description_column", "description")]
                        ),
                        "category": str(
                            product.get(schema.get("category_column", "category"), "")
                        ),
                        "score": score,
                        "text": " ".join(text_parts),
                        "metadata": {},
                    }

                    # Add custom fields
                    custom_fields = schema.get("custom_fields", [])
                    for field in custom_fields:
                        if field in product:
                            candidate["metadata"][field] = str(product[field])

                    candidates.append(candidate)

                except Exception as e:
                    logger.warning(f"Error processing candidate at index {idx}: {e}")
                    continue

            # Rerank candidates with context
            reranked_results = self.hybrid_search.rerank(
                query, candidates, query_analysis, top_k=max_items
            )

            return {
                "results": reranked_results,
                "total": len(reranked_results),
                "query_info": {
                    "original": query,
                    "analysis": query_analysis,
                    "model": {"name": model_name, "path": model_path},
                },
            }

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    def _load_products(self, model_path: str) -> Optional[pd.DataFrame]:
        local_path = os.path.join(AppConfig.BASE_MODEL_DIR, model_path, "products.csv")
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        return self.s3_manager.get_csv_content(f"{model_path}/products.csv")

    def _initialize_required_models(self):
        try:
            for model_key in AppConfig.REQUIRED_MODELS:
                logger.info(f"Preloading model: {model_key}")
                self.embedding_manager.get_model(model_key)
            logger.info("Successfully preloaded all required models")
        except Exception as e:
            logger.error(f"Error preloading models: {e}")


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


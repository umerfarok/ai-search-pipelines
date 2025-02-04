# search_service.py

from datetime import datetime
import os
import logging
import threading
import torch
import numpy as np
import json
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from flask import Flask, request, jsonify
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import pandas as pd
import io
from pathlib import Path
from collections import OrderedDict
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import spacy
from transformers import pipeline
from config import AppConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class QueryUnderstanding:
    def __init__(self):
        self.intent_classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )
        self.nlp = spacy.load("en_core_web_sm")
        self.domain_keywords = AppConfig.DOMAIN_KEYWORDS

    def analyze(self, query: str) -> Dict:
        try:
            intent = self.intent_classifier(
                query,
                candidate_labels=list(self.domain_keywords.keys()),
                multi_label=True,
            )

            doc = self.nlp(query)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            expanded_terms = []
            for label in intent.get("labels", []):
                if intent.get("scores", [0])[intent["labels"].index(label)] > 0.5:
                    expanded_terms.extend(self.domain_keywords.get(label, []))

            return {
                "intent": intent.get("labels", []),
                "intent_scores": intent.get("scores", []),
                "entities": entities,
                "expanded_terms": list(set(expanded_terms)),
                "original_query": query,
            }
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {
                "intent": [],
                "intent_scores": [],
                "entities": [],
                "expanded_terms": [],
                "original_query": query,
            }


class HybridSearchEngine:
    def __init__(self):
        self.bm25 = None
        self.tokenized_corpus = None
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _init_reranker(self):
        """Lazy initialization of reranker"""
        if self.reranker is None:
            try:
                self.reranker = CrossEncoder(self.reranker_name)
            except Exception as e:
                logger.error(f"Failed to initialize reranker: {e}")
                raise

    def initialize(self, texts: List[str]):
        """Initialize BM25 with corpus"""
        try:
            if not texts:
                raise ValueError("Empty text corpus")
                
            self.tokenized_corpus = [word_tokenize(text.lower()) for text in texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
        except Exception as e:
            logger.error(f"Failed to initialize BM25: {e}")
            raise

    def hybrid_search(
        self, query: str, semantic_scores: np.ndarray, alpha: float = 0.7
    ) -> np.ndarray:
        if not query.strip():
            return np.zeros_like(semantic_scores)

        tokenized_query = word_tokenize(query.lower())
        if not tokenized_query:
            return np.zeros_like(semantic_scores)

        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # Normalize scores
        bm25_range = bm25_scores.max() - bm25_scores.min()
        semantic_range = semantic_scores.max() - semantic_scores.min()

        if bm25_range > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / bm25_range
        if semantic_range > 0:
            semantic_scores = (semantic_scores - semantic_scores.min()) / semantic_range

        combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
        return combined_scores

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank results using cross-encoder with improved error handling"""
        if not candidates:
            return []

        try:
            self._init_reranker()
            
            pairs = [(query, doc["text"]) for doc in candidates]
            scores = self.reranker.predict(pairs)

            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)

            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original ordering if reranking fails
            return candidates[:top_k]


class EmbeddingManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_models = {}
        self.model_load_lock = threading.Lock()
        self.tokenizers = {}
        self.max_seq_length = 512
        self.cache_dir = AppConfig.TRANSFORMER_CACHE

        # Exact model mappings matching frontend configuration
        self.MODEL_MAPPINGS = self.MODEL_MAPPINGS = AppConfig.MODEL_MAPPINGS
        self.default_model_name = "all-minilm-l6"  # Matches frontend default

    def _get_model_info(self, model_name: str) -> dict:
        """Get model information with improved path matching"""
        # Check by key first
        if model_name in self.MODEL_MAPPINGS:
            return self.MODEL_MAPPINGS[model_name]
        
        # Check by path
        for key, info in self.MODEL_MAPPINGS.items():
            if info["path"] == model_name:
                return info
                
        # For direct paths that match our supported models
        for key, info in self.MODEL_MAPPINGS.items():
            if model_name.lower().replace("-", "") in info["path"].lower().replace("-", ""):
                return info

        raise ValueError(f"Unsupported model: {model_name}. Please use one of: {list(self.MODEL_MAPPINGS.keys())}")

    def get_model(self, model_name: str = None):
        if not model_name:
            model_name = self.default_model_name

        model_info = self._get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unsupported model: {model_name}")

        model_path = model_info["path"]
        
        if model_path in self.embedding_models:
            return self.embedding_models[model_path]

        try:
            with self.model_load_lock:
                if model_path not in self.embedding_models:
                    logger.info(f"Loading model: {model_path}")
                    model = SentenceTransformer(
                        model_path,
                        cache_folder=self.cache_dir,
                        device=self.device
                    )
                    
                    # Verify model dimension
                    actual_dim = model.get_sentence_embedding_dimension()
                    expected_dim = model_info["dimension"]
                    if actual_dim != expected_dim:
                        raise ValueError(
                            f"Model dimension mismatch for {model_name}: "
                            f"expected {expected_dim}, got {actual_dim}"
                        )
                    
                    self.embedding_models[model_path] = model
                    logger.info(f"Successfully loaded model: {model_path}")
                return self.embedding_models[model_path]

        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {model_path}") from e

    def get_model_dimension(self, model_name: str) -> int:
        """Get the embedding dimension for a model"""
        model_info = self._get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Cannot determine dimension for model: {model_name}")
        return model_info["dimension"]

    def generate_embedding(self, text: List[str], model_name: str = None) -> np.ndarray:
        """Generate embeddings with improved error handling"""
        if not text:
            raise ValueError("Empty text input")
            
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not initialized")

        # Dynamic batch size based on available memory
        batch_size = 512 if torch.cuda.is_available() else 128
        
        try:
            with torch.no_grad():
                embeddings = model.encode(
                    text,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device,
                    batch_size=batch_size,
                    normalize_embeddings=True
                )
                
                # Verify embedding dimension
                expected_dim = self.get_model_dimension(model_name)
                if embeddings.shape[1] != expected_dim:
                    raise ValueError(
                        f"Generated embedding dimension {embeddings.shape[1]} "
                        f"doesn't match expected {expected_dim}"
                    )
                
                return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if a model name is supported"""
        try:
            self._get_model_info(model_name)
            return True
        except ValueError:
            return False


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


class SearchService:
    def __init__(self):
        self.s3_manager = S3Manager()
        self.model_cache = ModelCache(AppConfig.MODEL_CACHE_SIZE)
        self.embedding_manager = EmbeddingManager()
        self.query_analyzer = QueryUnderstanding()
        self.hybrid_search = HybridSearchEngine()
        self._initialize_required_models()

    def _initialize_required_models(self):
        """Preload required models during initialization"""
        try:
            for model_key in AppConfig.REQUIRED_MODELS:
                logger.info(f"Preloading model: {model_key}")
                self.embedding_manager.get_model(model_key)
            logger.info("Successfully preloaded all required models")
        except Exception as e:
            logger.error(f"Error preloading models: {e}")

    def search(
        self, 
        query: str, 
        model_path: str, 
        max_items: int = 10, 
        alpha: float = 0.7,
        min_score: float = 0.0
    ) -> Dict:
        try:
            if not query.strip():
                return {
                    "results": [],
                    "total": 0,
                    "query_info": {
                        "original": query,
                        "enhanced": query,
                        "error": "Empty query",
                        "model_path": model_path
                    }
                }

            # Load model data
            model_data = self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            # Analyze and enhance query
            query_analysis = self.query_analyzer.analyze(query)
            enhanced_query = f"{query} {' '.join(query_analysis['expanded_terms'])}"

            # Get model name from metadata and verify dimensions
            model_name = model_data["metadata"]["models"]["embeddings"]
            stored_dimension = model_data["embeddings"].shape[1]
            expected_dimension = self.embedding_manager.get_model_dimension(model_name)
            
            if stored_dimension != expected_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: stored={stored_dimension}, "
                    f"expected={expected_dimension} for model {model_name}"
                )

            # Generate query embedding
            try:
                query_embed = self.embedding_manager.generate_embedding([enhanced_query], model_name)
                query_embedding = query_embed.reshape(-1)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Initialize hybrid search if needed
            if not self.hybrid_search.bm25:
                texts_path = os.path.join(AppConfig.BASE_MODEL_DIR, model_path, "processed_texts.json")
                if not os.path.exists(texts_path):
                    raise FileNotFoundError(f"Processed texts not found at {texts_path}")
                    
                with open(texts_path, "r") as f:
                    processed_texts = json.load(f)
                self.hybrid_search.initialize(processed_texts)

            # Calculate similarities
            semantic_scores = np.dot(model_data["embeddings"], query_embedding)
            combined_scores = self.hybrid_search.hybrid_search(
                enhanced_query,
                semantic_scores,
                alpha=alpha
            )

            # Get top candidates
            top_k_idx = np.argsort(combined_scores)[-max_items * 2:][::-1]

            # Load products
            products_df = self._load_products(model_path)
            if products_df is None:
                raise ValueError("Failed to load products data")

            # Prepare candidates for reranking
            schema = model_data["metadata"]["schema_mapping"]
            candidates = []

            # Map schema keys correctly
            schema_mapping = {
                "id": schema.get("id_column") or schema.get("idcolumn"),
                "name": schema.get("name_column") or schema.get("namecolumn"),
                "description": schema.get("description_column") or schema.get("descriptioncolumn"),
                "category": schema.get("category_column") or schema.get("categorycolumn")
            }

            for idx in top_k_idx:
                try:
                    score = float(combined_scores[idx])
                    if score < min_score:
                        continue

                    product = products_df.iloc[idx]

                    # Build text for ranking with fallback column names
                    text_parts = []
                    for col in [schema_mapping["name"], schema_mapping["description"], schema_mapping["category"]]:
                        if col in product:
                            text_parts.append(str(product[col]))
                        else:
                            logger.warning(f"Column {col} not found in product data")

                    candidate = {
                        "id": str(product[schema_mapping["id"]]),
                        "name": str(product[schema_mapping["name"]]),
                        "description": str(product[schema_mapping["description"]]),
                        "category": str(product[schema_mapping["category"]]),
                        "score": score,
                        "text": " ".join(text_parts),
                        "metadata": {}
                    }

                    # Add custom fields
                    custom_columns = schema.get("custom_columns", [])
                    for col in custom_columns:
                        if isinstance(col, dict) and "name" in col and col["name"] in product:
                            candidate["metadata"][col["name"]] = str(product[col["name"]])

                    candidates.append(candidate)

                except Exception as e:
                    logger.warning(f"Error processing candidate at index {idx}: {e}")
                    continue

            # Rerank candidates
            reranked_results = self.hybrid_search.rerank(
                enhanced_query,
                candidates,
                top_k=max_items
            )

            return {
                "results": reranked_results,
                "total": len(reranked_results),
                "query_info": {
                    "original": query,
                    "enhanced": enhanced_query,
                    "analysis": query_analysis,
                    "model_path": model_path,
                    "model": {
                        "name": model_name,
                        "dimension": expected_dimension
                    },
                    "schema": schema_mapping,  # Include schema mapping in response for debugging
                }
            }

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    def _load_products(self, model_path: str) -> Optional[pd.DataFrame]:
        local_path = os.path.join(AppConfig.BASE_MODEL_DIR, model_path, "products.csv")
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        return self.s3_manager.get_csv_content(f"{model_path}/products.csv")

    def load_model(self, model_path: str) -> Optional[Dict]:
        """Enhanced model loading with better metadata handling"""
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
        try:
            # Load embeddings
            embeddings_path = os.path.join(model_dir, "embeddings.npy")
            embeddings = np.load(embeddings_path)

            # Load and validate metadata
            with open(os.path.join(model_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)

            # Verify model configuration
            if "models" not in metadata or "embeddings" not in metadata["models"]:
                raise ValueError("Invalid metadata: missing model information")

            model_name = metadata["models"]["embeddings"]
            expected_dim = self.embedding_manager.get_model_dimension(model_name)
            
            if embeddings.shape[1] != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: stored={embeddings.shape[1]}, "
                    f"expected={expected_dim} for model {model_name}"
                )

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
        try:
            model_dir = os.path.join(AppConfig.BASE_MODEL_DIR, model_path)
            os.makedirs(model_dir, exist_ok=True)

            required_files = [
                "embeddings.npy",
                "metadata.json",
                "products.csv",
                "processed_texts.json"
            ]

            # Download all required files
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
        
        
        
search_service = SearchService()


def initialize_service():
    """Initialize search service with model preloading"""
    logger.info("Initializing search service...")

    try:
        # Setup cache directories and environment
        AppConfig.setup_cache_dirs()

        # Initialize the search service (this will preload models)

        search_service._initialize_required_models()

        # Download required NLTK data
        try:
            nltk.download("punkt")
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {e}")
            return False

        logger.info("Search service initialization completed")
        return True

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False


# Flask routes
@app.route("/search", methods=["POST"])
def search():
    """Enhanced search endpoint with hybrid search and query understanding"""
    try:
        data = request.get_json()
        if not data or "model_path" not in data or "query" not in data:
            return jsonify({"error": "Invalid request"}), 400

        results = search_service.search(
            query=data["query"],
            model_path=data["model_path"],
            max_items=data.get("max_items", 10),
            alpha=data.get("alpha", 0.7),
        )
        print(results)
        return jsonify(results)

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/models/<model_path>/info")
def get_model_info(model_path):
    """Get detailed information about a specific model"""
    try:
        model_data = search_service.load_model(model_path)
        if not model_data:
            return jsonify({"error": "Model not found"}), 404

        return jsonify(
            {
                "metadata": model_data["metadata"],
                "status": "loaded",
                "embedding_shape": model_data["embeddings"].shape,
                "last_loaded": model_data.get("loaded_at"),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    try:
        return jsonify(
            {
                "status": "healthy",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        )
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


def initialize_service():
    """Initialize search service"""
    logger.info("Initializing search service...")

    try:
        # Setup cache directories
        os.makedirs(AppConfig.BASE_MODEL_DIR, exist_ok=True)
        os.makedirs(AppConfig.TRANSFORMER_CACHE, exist_ok=True)

        # Configure NLTK data directory FIRST
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

        logger.info("Search service initialization completed")
        return True

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False


if __name__ == "__main__":
    print("Starting search service...")
    if not initialize_service():
        logger.error("Service initialization failed")
        exit(1)

    app.run(host="0.0.0.0", port=AppConfig.SERVICE_PORT)

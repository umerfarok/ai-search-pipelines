import os
import logging
import time
from typing import Dict, List, Optional, Union
import torch
import numpy as np
import json
from pathlib import Path
import redis
from datetime import datetime
import threading
from collections import OrderedDict
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer
from flask import Flask, request, jsonify
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import pandas as pd
import io
from enum import Enum
from rank_bm25 import BM25Okapi
import spacy
from transformers import pipeline
from nltk.tokenize import word_tokenize
import nltk
from config import AppConfig
from model_initializer import ModelInitializer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Modify spaCy initialization with better error handling
def initialize_nlp():
    """Initialize spaCy with fallback options"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    
    try:
        # Try importing directly first
        import en_core_web_sm
        return en_core_web_sm.load()
    except ImportError:
        try:
            logger.warning("SpaCy model not found. Attempting to download...")
            os.system("python -m spacy download en_core_web_sm")
            import en_core_web_sm
            return en_core_web_sm.load()
        except Exception as e:
            logger.error(f"Failed to initialize spaCy: {e}")
            # Return a minimal pipeline that won't break the application
            return spacy.blank("en")


# Replace direct spaCy loading with initialization function
nlp = initialize_nlp()


class ModelStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class QueryUnderstanding:
    def __init__(self):
        self.intent_classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )
        self.nlp = nlp  # Use the initialized nlp
        self.domain_keywords = {
            "cooking": [
                "cook",
                "recipe",
                "meal",
                "ingredient",
                "bake",
                "grill",
                "fry",
                "boil",
                "simmer",
                "spice",
            ],
            "tech": [
                "wifi",
                "laptop",
                "battery",
                "app",
                "smartphone",
                "tablet",
                "charger",
                "software",
                "hardware",
                "gadget",
            ],
            "health": [
                "headache",
                "remedy",
                "sleep",
                "sore",
                "exercise",
                "diet",
                "nutrition",
                "wellness",
                "therapy",
                "medicine",
            ],
            "household": [
                "clean",
                "tool",
                "fix",
                "repair",
                "maintenance",
                "appliance",
                "furniture",
                "decoration",
                "garden",
                "utility",
            ],
            "finance": [
                "investment",
                "stock",
                "bank",
                "loan",
                "credit",
                "insurance",
                "mortgage",
                "savings",
                "budget",
                "tax",
            ],
            "education": [
                "study",
                "course",
                "degree",
                "school",
                "university",
                "lecture",
                "homework",
                "exam",
                "teacher",
                "student",
            ],
            "travel": [
                "flight",
                "hotel",
                "tour",
                "destination",
                "cruise",
                "booking",
                "itinerary",
                "passport",
                "visa",
                "adventure",
            ],
            "sports": [
                "football",
                "basketball",
                "tennis",
                "cricket",
                "golf",
                "swimming",
                "athletics",
                "coach",
                "team",
                "tournament",
            ],
            "entertainment": [
                "movie",
                "music",
                "concert",
                "theater",
                "game",
                "festival",
                "celebrity",
                "show",
                "album",
                "ticket",
            ],
            "fashion": [
                "clothing",
                "style",
                "trend",
                "designer",
                "accessory",
                "outfit",
                "brand",
                "model",
                "runway",
                "boutique",
            ],
        }

    def analyze(self, query: str) -> Dict:
        try:
            # Fallback intent classification if NLP fails
            intent = {"labels": [], "scores": []}
            entities = []

            try:
                # Intent classification
                intent = self.intent_classifier(
                    query,
                    candidate_labels=list(self.domain_keywords.keys()),
                    multi_label=True,
                )

                # Entity recognition
                doc = self.nlp(query)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
            except Exception as e:
                logger.warning(f"Advanced query analysis failed: {e}")

            # Continue with basic keyword matching even if advanced analysis fails
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

    def initialize(self, texts: List[str]):
        """Initialize BM25 with corpus"""
        self.tokenized_corpus = [word_tokenize(text.lower()) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def hybrid_search(
        self, query: str, semantic_scores: np.ndarray, alpha: float = 0.7
    ) -> np.ndarray:
        """Combine BM25 and semantic search scores"""
        if not query.strip():
            return np.zeros_like(semantic_scores)

        tokenized_query = word_tokenize(query.lower())
        if not tokenized_query:
            return np.zeros_like(semantic_scores)

        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))

        # Safe normalization
        bm25_range = bm25_scores.max() - bm25_scores.min()
        semantic_range = semantic_scores.max() - semantic_scores.min()

        if bm25_range > 0:
            bm25_scores = (bm25_scores - bm25_scores.min()) / bm25_range
        if semantic_range > 0:
            semantic_scores = (semantic_scores - semantic_scores.min()) / semantic_range

        combined_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
        return combined_scores

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank results using cross-encoder"""
        if not candidates:
            return []

        pairs = [(query, doc["text"]) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # Add rerank scores to candidates
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        # Sort by rerank score
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


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

    def upload_file(self, local_path: str, s3_path: str) -> bool:
        try:
            content_type = {
                ".json": "application/json",
                ".npy": "application/octet-stream",
                ".csv": "text/csv",
            }.get(Path(local_path).suffix, "application/octet-stream")

            self.s3.upload_file(
                local_path,
                self.bucket,
                s3_path,
                ExtraArgs={"ContentType": content_type},
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False


class EmbeddingManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_models = {}
        self.model_load_lock = threading.Lock()
        self.tokenizers = {}
        self.max_seq_length = 512

    def get_model(self, model_name: str = None):
        if not model_name:
            model_name = AppConfig.DEFAULT_MODEL

        with self.model_load_lock:
            if model_name not in self.embedding_models:
                try:
                    model = SentenceTransformer(
                        model_name, truncate_dim=self.max_seq_length
                    )
                    model.to(self.device)
                    self.embedding_models[model_name] = model
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    return None
            return self.embedding_models[model_name]

    def get_tokenizer(self, model_name: str):
        """Get or load tokenizer for a model"""
        if model_name not in self.tokenizers:
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizers[model_name]

    def generate_embedding(self, text: List[str], model_name: str = None) -> np.ndarray:
        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not initialized")

        batch_size = 512 if torch.cuda.is_available() else 128
        with torch.no_grad():
            # Update encoding to use tokenized input
            embeddings = model.encode(
                text,
                convert_to_tensor=True,
                show_progress_bar=False,
                truncate_dim=512,
                device=self.device,
                batch_size=batch_size,
                normalize_embeddings=True,
            )

            return embeddings.cpu().numpy()


class ProductTrainer:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.s3_manager = S3Manager()
        self.domain_processors = {
            "food": self._enhance_food_text,
            "tech": self._enhance_tech_text,
            "general": self._enhance_general_text,
        }

    def _enhance_food_text(self, text: str, metadata: Dict) -> str:
        """Enhance food product text with domain-specific information"""
        parts = [text]
        if metadata.get("ingredients"):
            parts.append(f"Ingredients: {metadata['ingredients']}")
        if metadata.get("cook_time"):
            parts.append(f"Cooking time: {metadata['cook_time']}")
        return " ".join(parts)

    def _enhance_tech_text(self, text: str, metadata: Dict) -> str:
        """Enhance tech product text with specifications"""
        parts = [text]
        if metadata.get("specifications"):
            parts.append(f"Specs: {metadata['specifications']}")
        return " ".join(parts)

    def _enhance_general_text(self, text: str, metadata: Dict) -> str:
        """Default text enhancement"""
        return text

    def _process_text(self, row: pd.Series, schema: Dict, config: Dict) -> str:
        """Enhanced text processing with domain-specific handling"""
        # Basic text assembly
        text_parts = []

        # Process core fields
        for field in ["name", "description", "category"]:
            col = schema.get(f"{field}column")
            if col and col in row:
                text_parts.append(str(row[col]))

        # Join base text
        base_text = " ".join(filter(None, text_parts))

        # Get domain-specific processor
        domain = config.get("domain", "general")
        domain_processor = self.domain_processors.get(
            domain, self._enhance_general_text
        )

        # Create metadata dict for domain processing
        metadata = {}
        if "customcolumns" in schema:
            for col in schema["customcolumns"]:
                if col["name"] in row:
                    metadata[col["name"]] = str(row[col["name"]])

        # Apply domain-specific enhancements
        return domain_processor(base_text, metadata)

    def _load_data(self, config: Dict) -> Optional[pd.DataFrame]:
        try:
            data_source = config["data_source"]
            current_file = data_source["location"]

            df = self.s3_manager.get_csv_content(current_file)
            if df is None:
                raise ValueError(f"Failed to load CSV from {current_file}")

            # Handle append mode with previous version
            if config.get("mode") == "append" and config.get("previous_version"):
                prev_path = f"models/{config['previous_version']}/products.csv"
                prev_df = self.s3_manager.get_csv_content(prev_path)
                if prev_df is not None:
                    df = pd.concat([prev_df, df], ignore_index=True)

            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def train(self, job: Dict) -> bool:
        try:
            config = job
            config_id = config["_id"]["$oid"]

            df = self._load_data(config)
            if df is None or len(df) == 0:
                raise ValueError("No valid training data found")

            # Process texts with enhanced schema mapping
            processed_texts = [
                self._process_text(row, config["schema_mapping"], config)
                for _, row in df.iterrows()
            ]
            model_name = config["training_config"]["embeddingmodel"]
            embeddings_list = []
            total_rows = len(df)
            processed = 0
            chunk_size = 1000

            for i in range(0, total_rows, chunk_size):
                chunk_df = df.iloc[i : i + chunk_size]
                chunk_texts = [
                    self._process_text(row, config["schema_mapping"], config)
                    for _, row in chunk_df.iterrows()
                ]
                chunk_embeddings = self.embedding_manager.generate_embedding(
                    chunk_texts, model_name
                )
                embeddings_list.append(chunk_embeddings)

                processed += len(chunk_df)
                progress = (processed / total_rows) * 100
                logger.info(f"Processing progress: {progress:.2f}%")

            embeddings_array = np.vstack(embeddings_list)

            # Save and upload files
            model_path = f"models/{config_id}"
            model_dir = os.path.join(AppConfig.BASE_MODEL_DIR, model_path)

            if not self._save_model_files(
                model_dir, embeddings_array, df, config, processed_texts
            ):
                raise Exception("Failed to save model files")

            if not self._upload_model_files(model_dir, model_path):
                raise Exception("Failed to upload model files")

            return True

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False

    def _save_model_files(
        self,
        model_dir: str,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        config: Dict,
        processed_texts: List[str],
    ) -> bool:
        try:
            os.makedirs(model_dir, exist_ok=True)

            # Save embeddings and data
            np.save(os.path.join(model_dir, "embeddings.npy"), embeddings)
            df.to_csv(os.path.join(model_dir, "products.csv"), index=False)

            # Save processed texts for BM25
            with open(os.path.join(model_dir, "processed_texts.json"), "w") as f:
                json.dump(processed_texts, f)

            # Enhanced metadata with training configuration
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "num_samples": len(df),
                "embedding_shape": embeddings.shape,
                "models": {
                    "embeddings": config["training_config"]["embeddingmodel"],
                },
                "schema_mapping": config["schema_mapping"],
                "training_stats": {
                    "start_time": datetime.now().isoformat(),
                    "processed_records": len(df),
                    "total_records": len(df),
                },
            }

            with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error saving files: {e}")
            return False

    def _upload_model_files(self, local_dir: str, model_path: str) -> bool:
        try:
            files = [
                "embeddings.npy",
                "metadata.json",
                "products.csv",
                "processed_texts.json",
            ]

            for file in files:
                local_path = os.path.join(local_dir, file)
                s3_path = f"{model_path}/{file}"

                if not os.path.exists(local_path):
                    raise ValueError(f"Required file missing: {file}")

                if not self.s3_manager.upload_file(local_path, s3_path):
                    raise Exception(f"Failed to upload {file}")

            return True

        except Exception as e:
            logger.error(f"Error uploading files: {e}")
            return False


class SearchService:
    def __init__(self):
        self.s3_manager = S3Manager()
        self.model_cache = ModelCache(AppConfig.MODEL_CACHE_SIZE)
        self.embedding_manager = EmbeddingManager()
        self.query_analyzer = QueryUnderstanding()
        self.hybrid_search = HybridSearchEngine()

    def search(
        self, query: str, model_path: str, max_items: int = 10, alpha: float = 0.7
    ) -> Dict:
        try:
            # Load model data
            model_data = self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            # Analyze query
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            query_analysis = self.query_analyzer.analyze(query)
            enhanced_query = f"{query} {' '.join(query_analysis['expanded_terms'])}"

            # Get model name from metadata and generate query embedding
            model_name = model_data["metadata"]["config"]["training_config"][
                "embeddingmodel"
            ]
            try:
                query_embed = self.embedding_manager.generate_embedding(
                    [enhanced_query], model_name
                )
                query_embedding = (
                    query_embed.reshape(-1) if query_embed.ndim > 1 else query_embed
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Initialize hybrid search if not already done
            if not self.hybrid_search.bm25:
                with open(
                    os.path.join(
                        AppConfig.BASE_MODEL_DIR, model_path, "processed_texts.json"
                    ),
                    "r",
                ) as f:
                    processed_texts = json.load(f)
                self.hybrid_search.initialize(processed_texts)

            # Calculate semantic similarities
            semantic_scores = np.dot(model_data["embeddings"], query_embedding)

            # Get hybrid scores
            combined_scores = self.hybrid_search.hybrid_search(
                enhanced_query, semantic_scores, alpha=alpha
            )

            # Get top candidates
            top_k_idx = np.argsort(combined_scores)[-max_items * 2 :][
                ::-1
            ]  # Get 2x candidates for reranking

            # Load products
            products_df = self._load_products(model_path)
            if products_df is None:
                raise ValueError("Failed to load products data")

            # Prepare candidates for reranking
            schema = model_data["metadata"]["config"]["schema_mapping"]
            candidates = []

            for idx in top_k_idx:
                score = float(combined_scores[idx])
                if score < AppConfig.MIN_SCORE:
                    continue

                product = products_df.iloc[idx]

                # Create text representation for reranking
                text_parts = [
                    str(product[schema["namecolumn"]]),
                    str(product[schema["descriptioncolumn"]]),
                    str(product[schema["categorycolumn"]]),
                ]

                candidate = {
                    "id": str(product[schema["idcolumn"]]),
                    "name": str(product[schema["namecolumn"]]),
                    "description": str(product[schema["descriptioncolumn"]]),
                    "category": str(product[schema["categorycolumn"]]),
                    "score": score,
                    "text": " ".join(text_parts),
                    "metadata": {},
                }

                # Add custom fields
                if "customcolumns" in schema:
                    for col in schema["customcolumns"]:
                        if col.get("name") in product:
                            candidate["metadata"][col["name"]] = str(
                                product[col["name"]]
                            )

                candidates.append(candidate)

            # Rerank candidates
            reranked_results = self.hybrid_search.rerank(
                enhanced_query, candidates, top_k=max_items
            )

            return {
                "results": reranked_results,
                "total": len(reranked_results),
                "query_info": {
                    "original": query,
                    "enhanced": enhanced_query,
                    "analysis": query_analysis,
                    "model_path": model_path,
                },
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    def _load_products(self, model_path: str) -> Optional[pd.DataFrame]:
        local_path = os.path.join(AppConfig.BASE_MODEL_DIR, model_path, "products.csv")
        if os.path.exists(local_path):
            return pd.read_csv(local_path)
        return self.s3_manager.get_csv_content(f"{model_path}/products.csv")

    def load_model(self, model_path: str) -> Optional[Dict]:
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
            embeddings = np.load(os.path.join(model_dir, "embeddings.npy"))
            with open(os.path.join(model_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)

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

            files = [
                "embeddings.npy",
                "metadata.json",
                "products.csv",
                "processed_texts.json",
            ]
            for file in files:
                s3_path = f"{model_path}/{file}"
                local_path = os.path.join(model_dir, file)

                if not self.s3_manager.download_file(s3_path, local_path):
                    logger.error(f"Failed to download required file: {file}")
                    return None

            return self._load_from_local(model_dir, model_path)
        except Exception as e:
            logger.error(f"Error loading from S3: {e}")
            return None


class TrainingWorker:
    def __init__(self):
        self.trainer = ProductTrainer()
        self.redis = redis.Redis(
            host=AppConfig.REDIS_HOST,
            port=AppConfig.REDIS_PORT,
            password=AppConfig.REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=5,
            retry_on_timeout=True,
            socket_keepalive=True,
        )
        self.should_stop = False
        self.current_job = None
        self._worker_thread = None

    def process_job(self, job: Dict):
        """Process a single training job"""
        try:
            logger.info(f"Processing job for config: {job['_id']['$oid']}")
            config_id = job["_id"]["$oid"]

            self._update_status(config_id, ModelStatus.PROCESSING, progress=0)

            # Validate schema mapping
            if not self._validate_schema_mapping(job):
                raise ValueError("Invalid schema mapping configuration")

            success = self.trainer.train(job)

            if success:
                self._update_status(config_id, ModelStatus.COMPLETED, progress=100)
            else:
                self._update_status(
                    config_id, ModelStatus.FAILED, error="Training failed"
                )

        except Exception as e:
            logger.error(f"Error processing job: {e}")
            self._update_status(job["_id"]["$oid"], ModelStatus.FAILED, error=str(e))

    def _validate_schema_mapping(self, config: Dict) -> bool:
        """Validate schema mapping configuration"""
        required_fields = [
            "idcolumn",
            "namecolumn",
            "descriptioncolumn",
            "categorycolumn",
        ]
        schema = config.get("schema_mapping", {})

        # Check required fields
        if not all(field in schema for field in required_fields):
            return False

        # Validate custom columns if present
        if "customcolumns" in schema:
            for col in schema["customcolumns"]:
                if not all(field in col for field in ["name", "type", "role"]):
                    return False

        return True

    def _update_status(
        self,
        config_id: str,
        status: ModelStatus,
        progress: float = None,
        error: str = None,
    ):
        """Update job status in Redis"""
        try:
            status_data = {
                "status": status.value,
                "timestamp": datetime.now().isoformat(),
            }

            if progress is not None:
                status_data["training_stats"] = {
                    "progress": progress,
                    "processed_records": 0,
                    "total_records": 0,
                    "start_time": datetime.now().isoformat() if progress == 0 else None,
                    "end_time": datetime.now().isoformat() if progress == 100 else None,
                }
            if error:
                status_data["error"] = error

            key = f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}"
            self.redis.set(key, json.dumps(status_data), ex=86400)  # 24 hour expiry

        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def _worker_loop(self):
        """Main worker loop"""
        logger.info("Starting training worker loop")
        while not self.should_stop:
            try:
                result = self.redis.brpop(AppConfig.TRAINING_QUEUE, timeout=1)
                if result:
                    _, job_data = result
                    job = json.loads(job_data)
                    self.current_job = job
                    self.process_job(job)
                    self.current_job = None
                else:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)

    def start(self):
        """Start the training worker"""
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.should_stop = False
            self._worker_thread = threading.Thread(target=self._worker_loop)
            self._worker_thread.daemon = True
            self._worker_thread.start()
            logger.info("Training worker started")

    def stop(self):
        """Stop the training worker"""
        logger.info("Stopping training worker")
        self.should_stop = True
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            logger.info("Training worker stopped")


# Initialize services
search_service = SearchService()
worker = TrainingWorker()


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
        return jsonify(results)

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/models")
def get_available_models():
    """Get available embedding models info"""
    return jsonify(
        {
            "models": AppConfig.AVAILABLE_MODELS,
            "default": AppConfig.DEFAULT_MODEL,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )


@app.route("/models/update", methods=["POST"])
def update_available_models():
    """Update available models configuration"""
    try:
        data = request.get_json()
        if not data or "models" not in data:
            return jsonify({"error": "Invalid request"}), 400

        # Update models configuration
        AppConfig.AVAILABLE_MODELS.update(data["models"])
        if "default_model" in data:
            AppConfig.DEFAULT_MODEL = data["default_model"]

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    try:
        redis_ok = worker.redis.ping()
        return jsonify(
            {
                "status": "healthy",
                "redis": redis_ok,
                "worker": {
                    "running": worker._worker_thread is not None
                    and worker._worker_thread.is_alive(),
                    "current_job": (
                        worker.current_job["_id"]["$oid"]
                        if worker.current_job
                        else None
                    ),
                },
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        )
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/status/<config_id>")
def get_status(config_id):
    """Get training status"""
    try:
        status_key = f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}"
        status = worker.redis.get(status_key)

        if status:
            return jsonify(json.loads(status))

        if worker.current_job and worker.current_job["_id"]["$oid"] == config_id:
            return jsonify(
                {
                    "status": ModelStatus.PROCESSING.value,
                    "timestamp": datetime.now().isoformat(),
                }
            )
    except Exception as e:
        return jsonify({"error": "Status not found"}), 404


@app.route("/control/start", methods=["POST"])
def start_worker():
    """Start the training worker"""
    try:
        worker.start()
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/control/stop", methods=["POST"])
def stop_worker():
    """Stop the training worker"""
    try:
        worker.stop()
        return jsonify({"status": "stopped"})
    except Exception as e:
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


model_manager = ModelInitializer()


def initialize_service():
    """Initialize all service components"""
    logger.info("Initializing service...")

    # Setup cache directories
    AppConfig.setup_cache_dirs()

    # Initialize models
    if not model_manager.initialize_all():
        logger.error("Failed to initialize models")
        return False

    # Start worker
    worker.start()

    logger.info("Service initialization completed")
    return True


if __name__ == "__main__":
    if not initialize_service():
        logger.error("Service initialization failed")

    # Run Flask app
    app.run(host="0.0.0.0", port=AppConfig.SERVICE_PORT, use_reloader=False)

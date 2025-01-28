import os
import logging
from flask import Flask, request, jsonify
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
import redis
from typing import Dict, List, Optional, Tuple
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import pandas as pd
import io
from dataclasses import dataclass
from datetime import datetime
import threading
from collections import OrderedDict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from queue import Queue
import time
from model_initializer import ModelInitializer


# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@dataclass
class AppConfig:
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    MODEL_CACHE_SIZE: int = int(os.getenv("MODEL_CACHE_SIZE", 2))
    MIN_SCORE: float = float(os.getenv("MIN_SCORE", 0.2))
    AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY")
    S3_BUCKET: str = os.getenv("S3_BUCKET")
    S3_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ENDPOINT_URL: str = os.getenv("AWS_ENDPOINT_URL")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PRELOAD_MODELS: bool = os.getenv("PRELOAD_MODELS", "true").lower() == "true"
    MODEL_LOAD_THREADS: int = int(os.getenv("MODEL_LOAD_THREADS", 2))


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess_query(self, query: str) -> str:
        """Preprocess search query with enhanced understanding"""
        # Lowercase and tokenize
        tokens = word_tokenize(query.lower())

        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and token.isalnum()
        ]

        return " ".join(processed_tokens)

    def extract_intent(self, query: str) -> Dict[str, any]:
        """Extract search intent and potential filters from query"""
        tokens = word_tokenize(query.lower())
        pos_tags = nltk.pos_tag(tokens)

        intent = {"action": None, "object": None, "attributes": [], "filters": {}}

        for i, (token, pos) in enumerate(pos_tags):
            if pos.startswith("VB"):  # Verb
                intent["action"] = token
            elif pos.startswith("NN"):  # Noun
                if not intent["object"]:
                    intent["object"] = token
                else:
                    intent["attributes"].append(token)
            elif pos.startswith("JJ"):  # Adjective
                intent["attributes"].append(token)

        return intent


class S3Manager:
    def __init__(self):
        config = Config(retries=dict(max_attempts=3), s3={"addressing_style": "path"})

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AppConfig.AWS_ACCESS_KEY,
            aws_secret_access_key=AppConfig.AWS_SECRET_KEY,
            region_name=AppConfig.S3_REGION,
            endpoint_url=AppConfig.AWS_ENDPOINT_URL,
            config=config,
        )
        self.bucket = AppConfig.S3_BUCKET

    def download_file(self, s3_path: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3.download_file(self.bucket, s3_path, local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return False

    def get_csv_content(self, s3_path: str) -> Optional[pd.DataFrame]:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_path)
            return pd.read_csv(io.BytesIO(response["Body"].read()))
        except Exception as e:
            logger.error(f"Failed to read CSV from S3: {e}")
            return None


class ModelCache:
    def __init__(self, cache_size: int = 2):
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.load_queue = Queue()
        self.loading_threads = []

    def get(self, model_path: str) -> Optional[Dict]:
        with self.lock:
            if model_path in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(model_path)
                self.cache[model_path] = value
                return value
        return None

    def put(self, model_path: str, model_data: Dict):
        with self.lock:
            if model_path in self.cache:
                self.cache.pop(model_path)
            elif len(self.cache) >= self.cache_size:
                # Remove least recently used
                self.cache.popitem(last=False)
            self.cache[model_path] = model_data

    def start_loading_threads(self, num_threads: int):
        """Start background threads for model loading"""
        for _ in range(num_threads):
            thread = threading.Thread(target=self._model_loader_worker)
            thread.daemon = True
            thread.start()
            self.loading_threads.append(thread)

    def _model_loader_worker(self):
        """Background worker for loading models"""
        while True:
            try:
                model_path = self.load_queue.get()
                if model_path is None:
                    break

                if self.get(model_path) is None:  # Double check if still needed
                    search_service.load_model(model_path)

                self.load_queue.task_done()

            except Exception as e:
                logger.error(f"Error in model loader worker: {e}")

    def queue_model_for_loading(self, model_path: str):
        """Queue a model for background loading"""
        self.load_queue.put(model_path)


class SearchService:
    def __init__(self):
        self.s3_manager = S3Manager()
        self.model_cache = ModelCache(AppConfig.MODEL_CACHE_SIZE)
        self.text_processor = TextPreprocessor()
        self.device = AppConfig.DEVICE
        self.model_initializer = ModelInitializer()

        # Initialize required models on startup
        if not self.model_initializer.init_models():
            raise RuntimeError("Failed to initialize required models")

        # Start background loading threads
        if AppConfig.PRELOAD_MODELS:
            self.model_cache.start_loading_threads(AppConfig.MODEL_LOAD_THREADS)

        logger.info(f"Initialized SearchService using device: {self.device}")

    def load_model(self, model_path: str) -> Optional[Dict]:
        """Load model and its artifacts from S3"""
        try:
            # Check cache first
            cached_model = self.model_cache.get(model_path)
            if cached_model:
                return cached_model

            logger.info(f"Loading model from {model_path}")

            # Create temporary directory
            local_model_dir = f"/tmp/models/{Path(model_path).name}"
            os.makedirs(local_model_dir, exist_ok=True)

            # Download required files
            required_files = ["metadata.json", "model.pt", "embeddings.npy"]
            for file in required_files:
                s3_path = f"{model_path}/{file}"
                local_path = f"{local_model_dir}/{file}"
                if not self.s3_manager.download_file(s3_path, local_path):
                    raise FileNotFoundError(f"Failed to download {file}")

            # Load metadata
            with open(f"{local_model_dir}/metadata.json", "r") as f:
                metadata = json.load(f)

            # Get the base model name from metadata
            model_name = metadata["model_info"]["name"]

            # Ensure base model is downloaded
            if not self.model_initializer.ensure_model_downloaded(model_name):
                raise RuntimeError(f"Failed to download base model {model_name}")

            # Get local path for the base model
            base_model_path = self.model_initializer.get_model_path(model_name)
            if not base_model_path:
                raise RuntimeError(f"Base model path not found for {model_name}")

            # Initialize model from local path
            model = SentenceTransformer(base_model_path)
            model.to(self.device)

            # Load fine-tuned model state
            state_dict = torch.load(
                f"{local_model_dir}/model.pt", map_location=self.device
            )
            model.load_state_dict(state_dict)

            # Load embeddings
            embeddings = np.load(f"{local_model_dir}/embeddings.npy")

            # Cache model data
            model_data = {
                "model": model,
                "embeddings": embeddings,
                "metadata": metadata,
                "loaded_at": datetime.now().isoformat(),
            }

            self.model_cache.put(model_path, model_data)
            return model_data

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def semantic_search(
        self,
        query: str,
        model_data: Dict,
        max_items: int = 10,
        min_score: float = AppConfig.MIN_SCORE,
    ) -> List[Tuple[int, float]]:
        """Perform semantic search using the model"""
        model = model_data["model"]
        embeddings = model_data["embeddings"]

        # Preprocess query
        processed_query = self.text_processor.preprocess_query(query)

        # Generate query embedding
        query_embedding = (
            model.encode(
                processed_query, convert_to_tensor=True, normalize_embeddings=True
            )
            .cpu()
            .numpy()
        )

        # Calculate similarities
        similarities = np.dot(embeddings, query_embedding)

        # Get top k results above minimum score
        indices = np.where(similarities >= min_score)[0]
        top_k_idx = indices[np.argsort(similarities[indices])[-max_items:][::-1]]

        return [(idx, float(similarities[idx])) for idx in top_k_idx]

    def apply_filters(
        self, df: pd.DataFrame, filters: Dict, schema_mapping: Dict
    ) -> pd.DataFrame:
        """Apply filters to the DataFrame with enhanced filtering capabilities"""
        filtered_df = df.copy()

        try:
            for field, value in filters.items():
                # Handle standard columns
                if field in ["category", "name", "description"]:
                    column = schema_mapping.get(f"{field}_column")
                    if column and column in filtered_df.columns:
                        if isinstance(value, list):
                            filtered_df = filtered_df[filtered_df[column].isin(value)]
                        else:
                            filtered_df = filtered_df[
                                filtered_df[column].str.contains(
                                    str(value), case=False, na=False
                                )
                            ]

                # Handle custom columns
                else:
                    for custom_col in schema_mapping.get("custom_columns", []):
                        if custom_col["standard_column"] == field:
                            column = custom_col["user_column"]
                            if column in filtered_df.columns:
                                if isinstance(value, dict):
                                    # Range filter
                                    if "min" in value:
                                        filtered_df = filtered_df[
                                            filtered_df[column] >= value["min"]
                                        ]
                                    if "max" in value:
                                        filtered_df = filtered_df[
                                            filtered_df[column] <= value["max"]
                                        ]
                                elif isinstance(value, list):
                                    filtered_df = filtered_df[
                                        filtered_df[column].isin(value)
                                    ]
                                else:
                                    filtered_df = filtered_df[
                                        filtered_df[column]
                                        .astype(str)
                                        .str.contains(str(value), case=False, na=False)
                                    ]

        except Exception as e:
            logger.error(f"Error applying filters: {e}")

        return filtered_df

    def search(
        self,
        query: str,
        model_path: str,
        max_items: int = 10,
        filters: Optional[Dict] = None,
    ) -> Dict:
        """Perform semantic search with enhanced natural language understanding"""
        try:
            # Extract search intent
            intent = self.text_processor.extract_intent(query)

            # Load model
            model_data = self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            # Get schema and metadata
            metadata = model_data["metadata"]
            schema = metadata["config"]["schema_mapping"]

            # Load and filter products data
            products_path = f"{model_path}/data.csv"
            products_df = self.s3_manager.get_csv_content(products_path)
            if products_df is None:
                raise FileNotFoundError("Products data not found")

            # Apply intent-based and explicit filters
            intent_filters = {}
            if intent["attributes"]:
                # Add category filter based on extracted attributes
                if schema.get("category_column"):
                    intent_filters["category"] = intent["attributes"]

            # Combine intent filters with explicit filters
            combined_filters = {**(filters or {}), **intent_filters}
            if combined_filters:
                products_df = self.apply_filters(products_df, combined_filters, schema)

            # Perform semantic search
            search_results = self.semantic_search(
                query, model_data, max_items=max_items
            )

            # Prepare results
            results = []
            for idx, score in search_results:
                try:
                    if idx >= len(products_df):
                        continue

                    product = products_df.iloc[idx]
                    result = {
                        "id": str(product.get(schema["id_column"], f"item_{idx}")),
                        "name": str(product.get(schema["name_column"], "Unknown")),
                        "description": str(
                            product.get(schema["description_column"], "")
                        ),
                        "category": str(
                            product.get(schema["category_column"], "Uncategorized")
                        ),
                        "score": score,
                        "metadata": {},
                        "relevance": {
                            "intent_match": bool(
                                intent["object"]
                                and intent["object"]
                                in str(product.get(schema["name_column"], "")).lower()
                            ),
                            "attribute_match": any(
                                attr
                                in str(
                                    product.get(schema["description_column"], "")
                                ).lower()
                                for attr in intent["attributes"]
                            ),
                        },
                    }

                    # Add custom column metadata
                    for col in schema.get("custom_columns", []):
                        if col["role"] == "metadata" and col["user_column"] in product:
                            result["metadata"][col["standard_column"]] = str(
                                product[col["user_column"]]
                            )

                    results.append(result)

                except Exception as e:
                    logger.error(f"Error processing result at index {idx}: {e}")
                    continue

            return {
                "results": results,
                "total": len(results),
                "query_info": {
                    "original": query,
                    "processed": self.text_processor.preprocess_query(query),
                    "intent": intent,
                    "model_path": model_path,
                },
                "filters_applied": bool(combined_filters),
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise


class ModelPreloader:
    def __init__(self, search_service: SearchService):
        self.search_service = search_service
        self.s3_manager = search_service.s3_manager
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start the model preloader thread"""
        if not self._thread or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._preload_loop)
            self._thread.daemon = True
            self._thread.start()

    def stop(self):
        """Stop the model preloader thread"""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _preload_loop(self):
        """Main preloader loop"""
        while not self._stop_event.is_set():
            try:
                # List objects in models directory
                response = self.s3_manager.s3.list_objects_v2(
                    Bucket=self.s3_manager.bucket, Prefix="models/"
                )

                # Extract model paths
                model_paths = set()
                for obj in response.get("Contents", []):
                    path = obj["Key"].split("/")
                    if len(path) > 2:  # models/model_id/file
                        model_paths.add(f"models/{path[1]}")

                # Queue models for loading
                for path in model_paths:
                    if not self.search_service.model_cache.get(path):
                        self.search_service.model_cache.queue_model_for_loading(path)

                # Sleep before next check
                time.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in preloader loop: {e}")
                time.sleep(60)  # Wait before retry


# Initialize search service and preloader
search_service = SearchService()
model_preloader = ModelPreloader(search_service)


@app.route("/search", methods=["POST"])
def search():
    """Handle search requests"""
    try:
        data = request.get_json()
        if not data or "model_path" not in data or "query" not in data:
            return jsonify({"error": "Invalid request"}), 400

        results = search_service.search(
            query=data["query"],
            model_path=data["model_path"],
            max_items=data.get("max_items", 10),
            filters=data.get("filters", {}),
        )
        return jsonify(results)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Search error: {error_msg}")
        return jsonify({"error": error_msg}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "device": AppConfig.DEVICE,
            "cached_models": len(search_service.model_cache.cache),
            "preloader_active": bool(
                model_preloader._thread and model_preloader._thread.is_alive()
            ),
        }
    )


def startup():
    """Startup handler"""
    if AppConfig.PRELOAD_MODELS:
        model_preloader.start()


def cleanup():
    """Cleanup handler"""
    model_preloader.stop()


if __name__ == "__main__":
    # Register handlers
    import atexit

    atexit.register(cleanup)

    # Start preloader
    startup()

    # Run Flask app
    port = int(os.getenv("SERVICE_PORT", 5000))
    app.run(host="0.0.0.0", port=port)

import os
import logging
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
import redis
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import pandas as pd
import io
from dataclasses import dataclass
from datetime import datetime
import threading
from collections import OrderedDict
import onnxruntime as ort
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@dataclass
class AppConfig:
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    MODEL_CACHE_SIZE: int = int(
        os.getenv("MODEL_CACHE_SIZE", 4)
    )  # Increased cache size
    MIN_SCORE: float = float(os.getenv("MIN_SCORE", 0.2))
    AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY")
    S3_BUCKET: str = os.getenv("S3_BUCKET")
    S3_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_SSL_VERIFY: bool = os.getenv("AWS_SSL_VERIFY", "true").lower() == "true"
    AWS_ENDPOINT_URL: str = os.getenv("AWS_ENDPOINT_URL")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1

    # Updated model configurations for better efficiency
    AVAILABLE_LLM_MODELS = {
        "distilgpt2-product": {
            "name": "distilgpt2",
            "description": "Lightweight GPT-2 model for product search",
        },
        "all-minilm-l6": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "Efficient embedding model for semantic search",
        },
    }
    DEFAULT_MODEL = "distilgpt2-product"

    # Cache and storage paths
    SHARED_MODELS_DIR: str = os.getenv("SHARED_MODELS_DIR", "/app/models")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/app/models/cache")
    TRANSFORMER_CACHE: str = os.getenv(
        "TRANSFORMER_CACHE", "/app/model_cache/transformers"
    )
    HF_HOME: str = os.getenv("HF_HOME", "/app/model_cache/huggingface")
    ONNX_CACHE_DIR: str = os.getenv("ONNX_CACHE_DIR", "/app/models/onnx")

    @classmethod
    def setup_cache_dirs(cls):
        """Setup all required cache directories"""
        for directory in [
            cls.MODEL_CACHE_DIR,
            cls.TRANSFORMER_CACHE,
            cls.HF_HOME,
            cls.ONNX_CACHE_DIR,
            os.path.join(cls.HF_HOME, "datasets"),
        ]:
            os.makedirs(directory, exist_ok=True)

        # Set environment variables
        os.environ.update(
            {
                "TORCH_HOME": cls.MODEL_CACHE_DIR,
                "TRANSFORMERS_CACHE": cls.TRANSFORMER_CACHE,
                "HF_HOME": cls.HF_HOME,
                "HF_DATASETS_CACHE": os.path.join(cls.HF_HOME, "datasets"),
            }
        )


class ModelCache:
    def __init__(self, cache_size: int = 4):
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.access_counts = {}
        self.last_access = {}

    def get(self, model_path: str) -> Optional[Dict]:
        with self.lock:
            if model_path in self.cache:
                self.access_counts[model_path] = (
                    self.access_counts.get(model_path, 0) + 1
                )
                self.last_access[model_path] = time.time()
                value = self.cache.pop(model_path)
                self.cache[model_path] = value
                return value

            cache_path = os.path.join(AppConfig.MODEL_CACHE_DIR, model_path)
            if os.path.exists(cache_path):
                try:
                    data = self._load_from_cache(cache_path)
                    self.put(model_path, data)
                    return data
                except Exception as e:
                    logger.error(f"Error loading from cache: {e}")

            return None

    def put(self, model_path: str, data: Dict):
        with self.lock:
            if len(self.cache) >= self.cache_size:
                # Enhanced eviction strategy
                if model_path in self.cache:
                    del self.cache[model_path]
                else:
                    # Calculate score based on frequency and recency
                    scores = {}
                    current_time = time.time()
                    for path in self.cache:
                        frequency = self.access_counts.get(path, 0)
                        recency = current_time - self.last_access.get(path, 0)
                        scores[path] = frequency / (1 + recency)

                    # Remove item with lowest score
                    to_remove = min(scores.items(), key=lambda x: x[1])[0]
                    self.cache.pop(to_remove)

            self.cache[model_path] = data
            self.access_counts[model_path] = 1
            self.last_access[model_path] = time.time()

            # Save to cache directory
            cache_path = os.path.join(AppConfig.MODEL_CACHE_DIR, model_path)
            self._save_to_cache(cache_path, data)

    def _save_to_cache(self, cache_path: str, data: Dict):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # Save embeddings
        np.save(f"{cache_path}/embeddings.npy", data["embeddings"])

        # Save metadata
        with open(f"{cache_path}/metadata.json", "w") as f:
            json.dump(data["metadata"], f)

        # Save ONNX model if available
        if "onnx_model" in data:
            onnx_path = os.path.join(AppConfig.ONNX_CACHE_DIR, f"{cache_path}.onnx")
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            torch.onnx.save(data["onnx_model"], onnx_path)

    def _load_from_cache(self, cache_path: str) -> Dict:
        embeddings = np.load(f"{cache_path}/embeddings.npy")

        with open(f"{cache_path}/metadata.json", "r") as f:
            metadata = json.load(f)

        data = {
            "embeddings": embeddings,
            "metadata": metadata,
            "loaded_at": datetime.now().isoformat(),
        }

        # Try loading ONNX model
        onnx_path = os.path.join(AppConfig.ONNX_CACHE_DIR, f"{cache_path}.onnx")
        if os.path.exists(onnx_path):
            data["onnx_model"] = ort.InferenceSession(onnx_path)

        return data


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
            verify=AppConfig.AWS_SSL_VERIFY,
            config=retry_config,
        )
        self.bucket = AppConfig.S3_BUCKET

    def download_file(self, s3_path: str, local_path: str) -> bool:
        for attempt in range(AppConfig.MAX_RETRIES):
            try:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                self.s3.download_file(self.bucket, s3_path, local_path)
                return True
            except Exception as e:
                if attempt == AppConfig.MAX_RETRIES - 1:
                    logger.error(f"Failed to download {s3_path}: {e}")
                    return False
                time.sleep(AppConfig.RETRY_DELAY * (attempt + 1))
        return False

    def get_csv_content(self, s3_path: str) -> Optional[pd.DataFrame]:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_path)
            return pd.read_csv(io.BytesIO(response["Body"].read()))
        except Exception as e:
            logger.error(f"Failed to read CSV from S3: {e}")
            return None


class ProductSearchManager:
    def __init__(self):
        self.device = torch.device(AppConfig.DEVICE)
        self.tokenizer = None
        self.model = None
        self.embedding_model = None
        self.onnx_session = None
        self.s3_manager = S3Manager()
        self.model_load_lock = threading.Lock()

    def load_models(self, model_path: str) -> bool:
        with self.model_load_lock:
            try:
                shared_path = os.path.join(AppConfig.SHARED_MODELS_DIR, model_path)
                llm_path = os.path.join(shared_path, "llm")

                # Try loading ONNX model first
                onnx_path = os.path.join(AppConfig.ONNX_CACHE_DIR, f"{model_path}.onnx")
                if os.path.exists(onnx_path):
                    self.onnx_session = ort.InferenceSession(
                        onnx_path,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    logger.info(f"Loaded ONNX model from {onnx_path}")
                else:
                    # Fall back to regular model loading
                    logger.info(f"Loading models from {llm_path}")

                    if not os.path.exists(llm_path):
                        raise ValueError(f"Model not found: {llm_path}")

                    # Load tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
                    if not self.tokenizer.pad_token:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                    # Load model with optimizations
                    model_kwargs = {
                        "device_map": "auto",
                        "torch_dtype": (
                            torch.float16
                            if torch.cuda.is_available()
                            else torch.float32
                        ),
                        "low_cpu_mem_usage": True,
                    }

                    self.model = AutoModelForCausalLM.from_pretrained(
                        llm_path, **model_kwargs
                    )

                    # Try to optimize with torch.jit
                    try:
                        self.model = torch.jit.script(self.model)
                        logger.info("Successfully created TorchScript model")
                    except Exception as e:
                        logger.warning(f"Could not create TorchScript model: {e}")

                # Load embedding model
                self.embedding_model = SentenceTransformer(
                    AppConfig.AVAILABLE_LLM_MODELS["all-minilm-l6"]["name"]
                )
                self.embedding_model.to(self.device)

                return True

            except Exception as e:
                logger.error(f"Error loading models: {e}")
                return False

    def generate_response(self, query: str, product_context: str) -> str:
        try:
            if self.onnx_session:
                return self._generate_response_onnx(query, product_context)
            elif self.model and self.tokenizer:
                return self._generate_response_torch(query, product_context)
            else:
                raise ValueError("No model loaded")

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I couldn't generate a response about the products."

    def _generate_response_onnx(self, query: str, product_context: str) -> str:
        prompt = f"Query: {query}\nContext: {product_context}\nResponse: "

        # Prepare input
        inputs = self.tokenizer(
            prompt, return_tensors="np", truncation=True, max_length=512, padding=True
        )

        # Run inference
        outputs = self.onnx_session.run(
            None,
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )

        # Process output
        response = self.tokenizer.decode(outputs[0][0], skip_special_tokens=True)
        response = response.split("Response: ")[-1].strip()
        return (
            response
            if response
            else "I apologize, but I couldn't generate a meaningful response."
        )

    def _generate_response_torch(self, query: str, product_context: str) -> str:
        prompt = f"Query: {query}\nContext: {product_context}\nResponse: "

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    num_beams=4,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.split("Response: ")[-1].strip()

            return (
                response
                if response
                else "I apologize, but I couldn't generate a meaningful response."
            )

        except RuntimeError as e:
            logger.error(f"Runtime error in generate_response: {e}")
            return "I apologize, but there was an error processing your query."

    def generate_embedding(self, text: str) -> np.ndarray:
        try:
            with torch.no_grad():
                embedding = self.embedding_model.encode(
                    text, convert_to_tensor=True, show_progress_bar=False
                )
                return embedding.cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise


class SearchService:
    def __init__(self):
        self.s3_manager = S3Manager()
        self.model_cache = ModelCache(AppConfig.MODEL_CACHE_SIZE)
        self.device = AppConfig.DEVICE
        self.product_search = ProductSearchManager()
        self.preloaded_models = set()

    async def preload_models(self):
        """Preload frequently used models on startup"""
        try:
            # Get list of models from shared directory
            shared_models = [
                f
                for f in os.listdir(AppConfig.SHARED_MODELS_DIR)
                if os.path.isdir(os.path.join(AppConfig.SHARED_MODELS_DIR, f))
            ]

            for model_path in shared_models[: AppConfig.MODEL_CACHE_SIZE]:
                try:
                    await self.load_model(model_path)
                    self.preloaded_models.add(model_path)
                except Exception as e:
                    logger.error(f"Error preloading model {model_path}: {e}")

        except Exception as e:
            logger.error(f"Error during model preloading: {e}")

    async def load_model(self, model_path: str) -> Optional[Dict]:
        try:
            # Check cache first
            cached_model = self.model_cache.get(model_path)
            if cached_model:
                logger.info(f"Model {model_path} loaded from cache")
                return cached_model

            # Check shared storage
            shared_path = os.path.join(AppConfig.SHARED_MODELS_DIR, model_path)
            if os.path.exists(shared_path):
                model_data = await self._load_from_shared(shared_path)
                if model_data:
                    self.model_cache.put(model_path, model_data)
                    return model_data

            # Fall back to S3
            logger.info(f"Loading model {model_path} from S3")
            return await self._load_from_s3(model_path)

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    async def _load_from_shared(self, shared_path: str) -> Optional[Dict]:
        try:
            embeddings_path = os.path.join(shared_path, "embeddings.npy")
            metadata_path = os.path.join(shared_path, "metadata.json")

            if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
                return None

            embeddings = np.load(embeddings_path)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Initialize models if needed
            if not self.product_search.model:
                if not await self.product_search.load_models(shared_path):
                    raise ValueError("Failed to initialize models")

            return {
                "embeddings": embeddings,
                "metadata": metadata,
                "loaded_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error loading from shared storage: {e}")
            return None

    async def _load_from_s3(self, model_path: str) -> Optional[Dict]:
        try:
            # Download model files
            files_to_download = [
                "embeddings.npy",
                "metadata.json",
                "llm/config.json",
                "llm/pytorch_model.bin",
                "llm/tokenizer.json",
            ]

            shared_path = os.path.join(AppConfig.SHARED_MODELS_DIR, model_path)
            os.makedirs(shared_path, exist_ok=True)
            os.makedirs(os.path.join(shared_path, "llm"), exist_ok=True)

            for file in files_to_download:
                s3_path = f"{model_path}/{file}"
                local_path = os.path.join(shared_path, file)

                if not self.s3_manager.download_file(s3_path, local_path):
                    raise ValueError(f"Failed to download {file}")

            # Now load from shared storage
            return await self._load_from_shared(shared_path)

        except Exception as e:
            logger.error(f"Error loading from S3: {e}")
            return None

    async def search(self, query: str, model_path: str, max_items: int = 10) -> Dict:
        try:
            logger.info(f"Starting search with model: {model_path}")

            # Load model
            model_data = await self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            # Generate query embedding
            query_embedding = self.product_search.generate_embedding(query)
            embeddings = model_data["embeddings"]

            # Calculate similarities
            similarities = np.dot(embeddings, query_embedding)
            top_k_idx = np.argsort(similarities)[-max_items:][::-1]

            # Load products data
            products_df = await self._load_products(model_path)
            if products_df is None:
                raise ValueError("Failed to load products data")

            # Prepare results
            results = []
            schema = model_data["metadata"]["config"]["schema_mapping"]

            for idx in top_k_idx:
                score = float(similarities[idx])
                if score < AppConfig.MIN_SCORE:
                    continue

                product = products_df.iloc[idx]
                result = {
                    "id": str(product[schema["id_column"]]),
                    "name": str(product[schema["name_column"]]),
                    "description": str(product[schema["description_column"]]),
                    "category": str(product[schema["category_column"]]),
                    "score": score,
                    "metadata": {},
                }

                # Add custom metadata
                for col in schema.get("custom_columns", []):
                    if col["role"] == "metadata" and col["user_column"] in product:
                        result["metadata"][col["standard_column"]] = str(
                            product[col["user_column"]]
                        )

                results.append(result)

            # Generate natural language response
            if results:
                product_context = self._create_product_context(results[:3])
                response = self.product_search.generate_response(query, product_context)

                return {
                    "results": results,
                    "total": len(results),
                    "natural_response": response,
                    "query_info": {"original": query, "model_path": model_path},
                }

            return {
                "results": [],
                "total": 0,
                "natural_response": "No matching products found.",
                "query_info": {"original": query},
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    async def _load_products(self, model_path: str) -> Optional[pd.DataFrame]:
        # Try shared storage first
        shared_path = os.path.join(
            AppConfig.SHARED_MODELS_DIR, model_path, "products.csv"
        )
        if os.path.exists(shared_path):
            return pd.read_csv(shared_path)

        # Fall back to S3
        return self.s3_manager.get_csv_content(f"{model_path}/products.csv")

    def _create_product_context(self, products: List[Dict]) -> str:
        context = "Available products:\n\n"
        for idx, product in enumerate(products, 1):
            context += f"{idx}. {product['name']}\n"
            context += f"   Description: {product['description']}\n"
            context += f"   Category: {product['category']}\n"
            if product.get("metadata"):
                for key, value in product["metadata"].items():
                    context += f"   {key}: {value}\n"
            context += "\n"
        return context


# Initialize service
search_service = SearchService()
AppConfig.setup_cache_dirs()


@app.before_first_request
async def startup():
    """Initialize service on startup"""
    try:
        await search_service.preload_models()
    except Exception as e:
        logger.error(f"Error during startup: {e}")


@app.route("/search", methods=["POST"])
async def search():
    try:
        data = request.get_json()
        if not data or "model_path" not in data or "query" not in data:
            return jsonify({"error": "Invalid request"}), 400

        results = await search_service.search(
            query=data["query"],
            model_path=data["model_path"],
            max_items=data.get("max_items", 10),
        )
        return jsonify(results)

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "device": AppConfig.DEVICE,
            "cached_models": len(search_service.model_cache.cache),
            "preloaded_models": list(search_service.preloaded_models),
        }
    )
if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 5000))
    app.run(host="0.0.0.0", port=port)

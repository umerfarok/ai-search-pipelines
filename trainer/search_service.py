import os
import logging
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
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
from enum import Enum
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

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
    AWS_SSL_VERIFY: bool = os.getenv("AWS_SSL_VERIFY", "true").lower() == "true"
    AWS_ENDPOINT_URL: str = os.getenv("AWS_ENDPOINT_URL")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Updated model configurations for better product search
    AVAILABLE_LLM_MODELS = {
        "gpt2-product": {
            "name": "gpt2",  # We'll fine-tune this for product search
            "description": "Fine-tuned GPT-2 model for product search and recommendations",
        },
        "all-mpnet-base": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "description": "Powerful embedding model for semantic search",
        },
    }
    DEFAULT_MODEL = "gpt2-product"
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")
    TRANSFORMER_CACHE = os.getenv("TRANSFORMER_CACHE", "/app/model_cache/transformers")
    HF_HOME = os.getenv("HF_HOME", "/app/model_cache/huggingface")
    
    @classmethod
    def setup_cache_dirs(cls):
        """Setup cache directories for models"""
        os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.TRANSFORMER_CACHE, exist_ok=True)
        os.makedirs(cls.HF_HOME, exist_ok=True)
        
        # Set environment variables for model caching
        os.environ["TORCH_HOME"] = cls.MODEL_CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = cls.TRANSFORMER_CACHE
        os.environ["HF_HOME"] = cls.HF_HOME
        os.environ["HF_DATASETS_CACHE"] = os.path.join(cls.HF_HOME, "datasets")

    @classmethod
    def get_cache_path(cls, model_path: str) -> str:
        safe_path = model_path.replace("/", "_").replace("\\", "_")
        return os.path.join(cls.MODEL_CACHE_DIR, safe_path)



class S3Manager:
    def __init__(self):
        config = Config(retries=dict(max_attempts=3), s3={"addressing_style": "path"})

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AppConfig.AWS_ACCESS_KEY,
            aws_secret_access_key=AppConfig.AWS_SECRET_KEY,
            region_name=AppConfig.S3_REGION,
            endpoint_url=AppConfig.AWS_ENDPOINT_URL,
            verify=AppConfig.AWS_SSL_VERIFY,
            config=config,
        )
        self.bucket = AppConfig.S3_BUCKET
        logger.info(f"Initialized S3Manager with bucket: {self.bucket}")

    def download_file(self, s3_path: str, local_path: str) -> bool:
        try:
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    self.s3.download_file(self.bucket, s3_path, local_path)
                    logger.info(f"Downloaded s3://{self.bucket}/{s3_path} to {local_path}")
                    return True
                except ClientError as e:
                    retry_count += 1
                    error_code = e.response.get('Error', {}).get('Code')
                    if error_code == '404' or retry_count == max_retries:
                        logger.error(f"Failed to download from S3: {e}")
                        return False
                    time.sleep(1)  # Wait before retrying
                    
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return False

    def get_csv_content(self, s3_path: str) -> Optional[pd.DataFrame]:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_path)
            return pd.read_csv(io.BytesIO(response["Body"].read()))
        except ClientError as e:
            logger.error(f"Failed to read CSV from S3: {e}")
            return None


class ModelCache:
    def __init__(self, cache_size: int = 2):
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        os.makedirs(AppConfig.MODEL_CACHE_DIR, exist_ok=True)

    def get_cache_path(self, model_path: str) -> str:
        return AppConfig.get_cache_path(model_path)
        
    def get(self, model_path: str) -> Optional[Dict]:
        with self.lock:
            cache_dir = self.get_cache_path(model_path)
            if os.path.exists(f"{cache_dir}/embeddings.npy"):
                if model_path in self.cache:
                    # Move to end (most recently used)
                    value = self.cache.pop(model_path)
                    self.cache[model_path] = value
                    return value
                    
                # Load from disk cache
                embeddings = np.load(f"{cache_dir}/embeddings.npy")
                with open(f"{cache_dir}/metadata.json", "r") as f:
                    metadata = json.load(f)
                    
                model_data = {
                    "embeddings": embeddings,
                    "metadata": metadata,
                    "loaded_at": datetime.now().isoformat(),
                }
                self.put(model_path, model_data)
                return model_data
            return None

    def put(self, model_path: str, model_data: Dict):
        with self.lock:
            cache_dir = self.get_cache_path(model_path)
            os.makedirs(cache_dir, exist_ok=True)
            
            # Save to disk cache
            np.save(f"{cache_dir}/embeddings.npy", model_data["embeddings"])
            with open(f"{cache_dir}/metadata.json", "w") as f:
                json.dump(model_data["metadata"], f)
            
            # Update memory cache
            if model_path in self.cache:
                self.cache.pop(model_path)
            elif len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[model_path] = model_data

class ProductSearchManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_tokenizer = None
        self.llm_model = None
        self.embedding_model = None
        self.s3_manager = S3Manager()

    def load_models(self, model_path: str) -> bool:
        try:
            # Ensure model path exists in cache
            cache_path = os.path.join(AppConfig.MODEL_CACHE_DIR, model_path)
            llm_path = os.path.join(cache_path, "llm")
            
            if not os.path.exists(llm_path):
                raise ValueError(f"Model not found in cache: {llm_path}")

            # Load the fine-tuned LLM model
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            # Load embedding model
            self.embedding_model = SentenceTransformer(
                AppConfig.AVAILABLE_LLM_MODELS["all-mpnet-base"]["name"]
            )
            self.embedding_model.to(self.device)

            logger.info(f"Successfully loaded models from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def generate_response(self, query: str, product_context: str) -> str:
        try:
            if not self.llm_model or not self.llm_tokenizer:
                raise ValueError("Models not initialized")

            prompt = f"""Question: {query}
Context: {product_context}
Answer: """

            inputs = self.llm_tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )

            response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("Answer: ")[-1].strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I couldn't generate a response about the products."

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using the sentence-transformer model"""
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
        logger.info(f"Initialized SearchService using device: {self.device}")

    def load_model(self, model_path: str) -> Optional[Dict]:
        """Load model from cache or S3"""
        try:
            # First try to get from cache
            model_data = self.model_cache.get(model_path)
            if (model_data):
                logger.info(f"Found model {model_path} in cache")
                return model_data

            # If not in cache, download from S3
            cache_dir = self.model_cache.get_cache_path(model_path)
            os.makedirs(cache_dir, exist_ok=True)

            # Download required files from S3
            required_files = {
                f"{model_path}/metadata.json": f"{cache_dir}/metadata.json",
                f"{model_path}/embeddings.npy": f"{cache_dir}/embeddings.npy",
                f"{model_path}/products.csv": f"{cache_dir}/products.csv"
            }

            # Download LLM files with strict checking
            required_llm_files = [
                "config.json",
                "pytorch_model.bin",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json"
            ]

            for file in required_llm_files:
                s3_path = f"{model_path}/llm/{file}"
                local_path = f"{llm_dir}/{file}"
                logger.info(f"Downloading {s3_path} to {local_path}")
                if not os.path.exists(local_path):
                    if not self.s3_manager.download_file(s3_path, local_path):
                        raise Exception(f"Failed to download {s3_path}")

            # Verify all required files exist
            for file in required_llm_files:
                local_path = f"{llm_dir}/{file}"
                if not os.path.exists(local_path):
                    raise Exception(f"Required file {file} not found after download")

            # Download all files
            for s3_path, local_path in required_files.items():
                if not os.path.exists(local_path):
                    logger.info(f"Downloading {s3_path} to {local_path}")
                    if not self.s3_manager.download_file(s3_path, local_path):
                        raise Exception(f"Failed to download {s3_path}")

            # Load metadata
            with open(f"{cache_dir}/metadata.json", "r") as f:
                metadata = json.load(f)

            # Load embeddings
            embeddings = np.load(f"{cache_dir}/embeddings.npy")

            # Create model data
            model_data = {
                "embeddings": embeddings,
                "metadata": metadata,
                "loaded_at": datetime.now().isoformat(),
            }

            # Store in cache
            self.model_cache.put(model_path, model_data)

            # Initialize models
            if not self.product_search.llm_model:
                self.product_search.load_models(model_path)

            return model_data

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def search(
        self,
        query: str,
        model_path: str,
        max_items: int = 10,
        filters: Optional[Dict] = None,
    ) -> Dict:
        try:
            # Ensure model is loaded
            model_data = self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            # Now load the models if not already loaded
            if not self.product_search.llm_model:
                self.product_search.load_models(model_path)

            # Generate query embedding
            query_embedding = self.product_search.generate_embedding(query)

            # Get cached embeddings
            model_data = self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            embeddings = model_data["embeddings"]

            # Calculate similarities
            similarities = np.dot(embeddings, query_embedding)
            top_k_idx = np.argsort(similarities)[-max_items:][::-1]

            # Load products data
            products_df = self.s3_manager.get_csv_content(f"{model_path}/products.csv")
            if products_df is None:
                raise FileNotFoundError("Products data not found")

            # Apply filters if provided
            if filters:
                products_df = self.apply_filters(
                    products_df,
                    filters,
                    model_data["metadata"]["config"]["schema_mapping"],
                )

            # Prepare results
            results = []
            schema = model_data["metadata"]["config"]["schema_mapping"]

            for idx in top_k_idx:
                score = float(similarities[idx])
                if score < AppConfig.MIN_SCORE:
                    continue

                product = products_df.iloc[idx]
                result = {
                    "id": str(product.get(schema["id_column"], f"item_{idx}")),
                    "name": str(product.get(schema["name_column"], "Unknown")),
                    "description": str(product.get(schema["description_column"], "")),
                    "category": str(
                        product.get(schema["category_column"], "Uncategorized")
                    ),
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

            # Generate natural language response for top results
            if results:
                product_context = self._create_product_context(results[:3])
                natural_response = self.product_search.generate_response(
                    query, product_context
                )

                return {
                    "results": results,
                    "total": len(results),
                    "natural_response": natural_response,
                    "query_info": {
                        "original": query,
                        "model_path": model_path,
                        "schema": schema,
                    },
                }

            return {
                "results": [],
                "total": 0,
                "natural_response": "I couldn't find any products matching your query.",
                "query_info": {"original": query},
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    def _create_product_context(self, products: List[Dict]) -> str:
        """Create detailed product context for LLM response generation"""
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


# Initialize the service
search_service = SearchService()
AppConfig.setup_cache_dirs()


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        if not data or "model_path" not in data or "query" not in data:
            return jsonify({"error": "Invalid request"}), 400

        model_path = data["model_path"]
        query = data["query"]
        max_items = data.get("max_items", 10)
        filters = data.get("filters", {})

        logger.info(f"Processing search request for model: {model_path}")
        logger.info(f"Query: {query}")

        results = search_service.search(
            query=query, model_path=model_path, max_items=max_items, filters=filters
        )
        return jsonify(results)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Search error: {error_msg}")
        return jsonify({"error": error_msg}), 500


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "device": AppConfig.DEVICE,
            "cached_models": len(search_service.model_cache.cache),
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 5000))
    app.run(host="0.0.0.0", port=port)

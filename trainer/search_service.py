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
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from peft import PeftModel, PeftConfig
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@dataclass
class AppConfig:
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    MODEL_CACHE_SIZE: int = int(os.getenv("MODEL_CACHE_SIZE", 4))
    MIN_SCORE: float = float(os.getenv("MIN_SCORE", 0.2))
    AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY")
    S3_BUCKET: str = os.getenv("S3_BUCKET")
    S3_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_SSL_VERIFY: bool = os.getenv("AWS_SSL_VERIFY", "true").lower() == "true"
    AWS_ENDPOINT_URL: str = os.getenv("AWS_ENDPOINT_URL")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 3))
    RETRY_DELAY: int = int(os.getenv("RETRY_DELAY", 1))

    # Model configurations
    AVAILABLE_LLM_MODELS = {
        "distilgpt2-product": {
            "name": "distilgpt2",
            "description": "Fast, lightweight model for product search",
            "peft_supported": True,
        },
        "all-minilm-l6": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "Efficient embedding model for semantic search",
            "peft_supported": False,
        },
    }
    DEFAULT_MODEL = "distilgpt2-product"

    # Simplified directory structure
    BASE_MODEL_DIR: str = os.getenv("BASE_MODEL_DIR", "/app/models")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")
    TRANSFORMER_CACHE: str = os.getenv(
        "TRANSFORMER_CACHE", "/app/model_cache/transformers"
    )
    HF_HOME: str = os.getenv("HF_HOME", "/app/model_cache/huggingface")

    @classmethod
    def setup_cache_dirs(cls):
        """Setup cache directories"""
        directories = [
            cls.BASE_MODEL_DIR,
            cls.MODEL_CACHE_DIR,
            cls.TRANSFORMER_CACHE,
            cls.HF_HOME,
            os.path.join(cls.HF_HOME, "datasets"),
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        os.environ.update(
            {
                "TORCH_HOME": cls.MODEL_CACHE_DIR,
                "TRANSFORMERS_CACHE": cls.TRANSFORMER_CACHE,
                "HF_HOME": cls.HF_HOME,
                "HF_DATASETS_CACHE": os.path.join(cls.HF_HOME, "datasets"),
            }
        )

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
        self.base_model = None
        self.peft_model = None
        self.embedding_model = None
        self.onnx_session = None
        self.model_load_lock = threading.Lock()

    def load_models(self, model_path: str) -> bool:
        """Load models with simplified path structure"""
        with self.model_load_lock:
            try:
                model_dir = os.path.join(AppConfig.BASE_MODEL_DIR, model_path)
                llm_dir = os.path.join(model_dir, "llm")
                logger.info(f"Loading models from {model_dir}")

                # Load metadata
                metadata_path = os.path.join(model_dir, "metadata.json")
                if not os.path.exists(metadata_path):
                    raise ValueError(f"Metadata not found in {model_dir}")

                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                # Verify PEFT configuration and adapter file
                peft_config_path = os.path.join(llm_dir, "adapter_config.json")
                peft_model_path = os.path.join(llm_dir, "adapter_model.bin")

                is_peft_model = os.path.exists(peft_config_path) and os.path.exists(
                    peft_model_path
                )

                # Initialize tokenizer
                tokenizer_path = os.path.join(llm_dir, "tokenizer.json")
                if not os.path.exists(tokenizer_path):
                    raise ValueError("Tokenizer not found")

                self.tokenizer = AutoTokenizer.from_pretrained(llm_dir)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.padding_side = "left"

                # Load models
                base_model_name = metadata.get("models", {}).get("llm", "distilgpt2")
                model_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": (
                        torch.float16 if torch.cuda.is_available() else torch.float32
                    ),
                    "low_cpu_mem_usage": True,
                }

                if is_peft_model:
                    logger.info("Loading PEFT model...")
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name, **model_kwargs
                    )

                    self.peft_model = PeftModel.from_pretrained(
                        self.base_model,
                        llm_dir,
                        is_trainable=False,
                        torch_dtype=model_kwargs["torch_dtype"],
                    )
                    self.peft_model.eval()
                else:
                    logger.info("Loading standard model...")
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        llm_dir, **model_kwargs
                    )
                    self.base_model.eval()

                # Load embedding model
                embedding_model_name = metadata.get("models", {}).get(
                    "embeddings", "sentence-transformers/all-MiniLM-L6-v2"
                )
                self.embedding_model = SentenceTransformer(embedding_model_name)
                self.embedding_model.to(self.device)

                logger.info(f"Successfully loaded models: PEFT={is_peft_model}")
                return True

            except Exception as e:
                logger.error(f"Failed to load model {model_path}. "
                 f"Existing files: {os.listdir(model_dir) if os.path.exists(model_dir) else 'Missing directory'}")
                raise

    def generate_response(self, query: str, product_context: str) -> str:
        """Generate response using appropriate model"""
        try:
            if self.peft_model:
                return self._generate_response_peft(query, product_context)
            elif self.base_model:
                return self._generate_response_base(query, product_context)
            else:
                raise ValueError("No model available for generation")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I couldn't process your query."
        
        
        
    def _generate_response_base(self, query: str, product_context: str) -> str:
        """Generate response using base model without PEFT"""
        try:
            prompt = f"Query: {query}\nContext: {product_context}\nResponse: "
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.base_model.generate(
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
            return response if response else "I couldn't generate a meaningful response."

        except Exception as e:
            logger.error(f"Error in base model response generation: {e}")
            return "I apologize, but there was an error processing your query."
    def _generate_response_peft(self, query: str, product_context: str) -> str:
        """Generate response using PEFT model"""
        try:
            prompt = f"Query: {query}\nContext: {product_context}\nResponse: "
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.peft_model.generate(
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
                response if response else "I couldn't generate a meaningful response."
            )

        except Exception as e:
            logger.error(f"Error in PEFT response generation: {e}")
            return "I apologize, but there was an error processing your query."

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using the sentence transformer model"""
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

    def load_model(self, model_path: str) -> Optional[Dict]:
        """Load model with simplified directory structure"""
        try:
            # Try loading from cache first
            cached_data = self.model_cache.get(model_path)
            if cached_data:
                return cached_data

            # Load from local directory
            model_dir = os.path.join(AppConfig.BASE_MODEL_DIR, model_path)
            if os.path.exists(model_dir):
                return self._load_from_local(model_dir, model_path)

            # Fall back to S3
            return self._load_from_s3(model_path)

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _load_from_local(self, model_dir: str, model_path: str) -> Optional[Dict]:
        """Load model from local directory"""
        try:
            # Load embeddings and metadata
            embeddings = np.load(os.path.join(model_dir, "embeddings.npy"))

            with open(os.path.join(model_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)

            # Initialize models
            if not self.product_search.load_models(model_path):
                raise ValueError("Failed to initialize models")

            data = {
                "embeddings": embeddings,
                "metadata": metadata,
                "loaded_at": datetime.now().isoformat(),
            }

            # Cache the loaded data
            self.model_cache.put(model_path, data)
            return data

        except Exception as e:
            logger.error(f"Error loading from local: {e}")
            return None

    def _load_from_s3(self, model_path: str) -> Optional[Dict]:
        """Load model from S3"""
        try:
            model_dir = os.path.join(AppConfig.BASE_MODEL_DIR, model_path)
            os.makedirs(os.path.join(model_dir, "llm"), exist_ok=True)

            # Download required files
            files_to_download = [
                "embeddings.npy",
                "metadata.json",
                "products.csv",
                "llm/adapter_config.json",
                "llm/adapter_model.bin",
                "llm/tokenizer.json",
                "llm/special_tokens_map.json",
                "llm/tokenizer_config.json",
            ]

            for file in files_to_download:
                s3_path = f"{model_path}/{file}"
                local_path = os.path.join(model_dir, file)

                if not self.s3_manager.download_file(s3_path, local_path):
                    logger.error(f"Failed to download required file: {file}")
                    return None

            # Load the downloaded model
            return self._load_from_local(model_dir, model_path)

        except Exception as e:
            logger.error(f"Error loading from S3: {e}")
            return None

    def search(self, query: str, model_path: str, max_items: int = 10) -> Dict:
        """Perform semantic search with response generation"""
        try:
            # Load model data
            model_data = self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            # Generate query embedding
            query_embedding = self.product_search.generate_embedding(query)

            # Calculate similarities
            similarities = np.dot(model_data["embeddings"], query_embedding)
            top_k_idx = np.argsort(similarities)[-max_items:][::-1]

            # Load products
            products_df = self._load_products(model_path)
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

            # Generate response
            if results:
                product_context = self._create_product_context(results[:3])
                response = self.product_search.generate_response(query, product_context)
            else:
                response = "No matching products found."

            return {
                "results": results,
                "total": len(results),
                "natural_response": response,
                "query_info": {
                    "original": query,
                    "model_path": model_path,
                },
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    def _load_products(self, model_path: str) -> Optional[pd.DataFrame]:
        """Load products data from local storage or S3"""
        # Try local storage first
        local_path = os.path.join(AppConfig.BASE_MODEL_DIR, model_path, "products.csv")
        if os.path.exists(local_path):
            return pd.read_csv(local_path)

        # Fall back to S3
        return self.s3_manager.get_csv_content(f"{model_path}/products.csv")

    def _create_product_context(self, products: List[Dict]) -> str:
        """Create context string from product information"""
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
        )
        return jsonify(results)

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "device": AppConfig.DEVICE,
            "model_dir": AppConfig.BASE_MODEL_DIR,
            "model_types": {
                "peft_available": search_service.product_search.peft_model is not None,
                "base_available": search_service.product_search.base_model is not None,
            },
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 5000))
    app.run(host="0.0.0.0", port=port)

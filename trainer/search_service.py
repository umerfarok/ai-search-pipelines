import os
import logging
import time
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
    SHARED_MODELS_DIR: str = os.getenv("SHARED_MODELS_DIR", "/app/models")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/app/models/cache")
    TRANSFORMER_CACHE = os.getenv("TRANSFORMER_CACHE", "/app/model_cache/transformers")
    HF_HOME = os.getenv("HF_HOME", "/app/model_cache/huggingface")

    SHARED_VOLUME_PATH = "/tmp/localstack-s3-storage/local-bucket"
    
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

    @classmethod
    def get_shared_model_path(cls, model_path: str) -> str:
        """Get path to model in shared volume"""
        return os.path.join(cls.SHARED_VOLUME_PATH, model_path)

    @classmethod
    def get_model_cache_path(cls, model_path: str) -> str:
        """Get full path to model in cache directory"""
        return os.path.join(cls.MODEL_CACHE_DIR, model_path)

    @classmethod
    def ensure_paths_exist(cls, model_path: str):
        """Ensure all required model paths exist"""
        cache_dir = cls.get_model_cache_path(model_path)
        llm_dir = os.path.join(cache_dir, "llm")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(llm_dir, exist_ok=True)
        return cache_dir, llm_dir

    @classmethod
    def get_model_s3_path(cls, model_path: str) -> str:
        """Normalize S3 path for model files"""
        # Remove 's3://' prefix if present
        model_path = model_path.replace('s3://', '')
        # Remove bucket name if present
        if model_path.startswith(f"{cls.S3_BUCKET}/"):
            model_path = model_path[len(cls.S3_BUCKET)+1:]
        return model_path

    @classmethod
    def ensure_model_files(cls, model_path: str, s3_manager: 'S3Manager') -> bool:
        """Ensure model files exist in shared directory"""
        try:
            # Get normalized paths
            shared_path = cls.get_model_path(model_path)
            llm_path = os.path.join(shared_path, "llm")
            os.makedirs(llm_path, exist_ok=True)
            
            logger.info(f"Checking model files in {shared_path}")

            # Define required files
            required_files = {
                "root": ["metadata.json", "embeddings.npy", "products.csv"],
                "llm": ["config.json", "pytorch_model.bin", "tokenizer.json"]
            }
            
            # First check shared directory
            all_files_exist = True
            for location, files in required_files.items():
                target_dir = llm_path if location == "llm" else shared_path
                for file in files:
                    file_path = os.path.join(target_dir, file)
                    if not os.path.exists(file_path):
                        all_files_exist = False
                        break

            # If files missing, try S3
            if not all_files_exist:
                logger.info("Some files missing, downloading from S3...")
                # Strip any models/ prefix for S3 path
                s3_model_path = model_path.replace('models/', '')
                
                for location, files in required_files.items():
                    target_dir = llm_path if location == "llm" else shared_path
                    for file in files:
                        file_path = os.path.join(target_dir, file)
                        if not os.path.exists(file_path):
                            s3_path = f"models/{s3_model_path}/{'llm/' if location == 'llm' else ''}{file}"
                            if not s3_manager.download_file(s3_path, file_path):
                                logger.error(f"Failed to download {s3_path}")
                                return False
            
            return True

        except Exception as e:
            logger.error(f"Error ensuring model files: {e}")
            return False

    @classmethod
    def get_model_path(cls, model_id: str) -> str:
        """Get full path to model in shared directory"""
        # Remove any leading slashes and 'models/' prefix for consistency
        model_id = model_id.lstrip('/').replace('models/', '')
        return os.path.join(cls.SHARED_MODELS_DIR, model_id)


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
                if (model_path in self.cache):
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
        # Initialize embedding model right away
        self.initialize_embedding_model()
        
    def initialize_embedding_model(self) -> None:
        """Initialize the embedding model separately"""
        try:
            logger.info("Initializing embedding model...")
            self.embedding_model = SentenceTransformer(
                AppConfig.AVAILABLE_LLM_MODELS["all-mpnet-base"]["name"]
            )
            self.embedding_model.to(self.device)
            logger.info("Successfully initialized embedding model")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise

    def load_models(self, model_path: str) -> bool:
        try:
            # Use shared model path directly
            shared_path = AppConfig.get_model_path(model_path)
            llm_path = os.path.join(shared_path, "llm")
            
            logger.info(f"Loading model from shared path: {llm_path}")
            
            if not os.path.exists(llm_path):
                raise ValueError(f"Model not found in shared path: {llm_path}")
            
            # Verify required files exist
            required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
            missing_files = [f for f in required_files 
                           if not os.path.exists(os.path.join(llm_path, f))]
            
            if missing_files:
                raise ValueError(f"Missing required model files in cache: {missing_files}")

            logger.info(f"Loading model from cache: {llm_path}")

            # Load the fine-tuned LLM model with additional safeguards
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
            
            # Set padding token if not set
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # Load model with safer defaults
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True,
                "use_safetensors": False  # Disable safetensors for better compatibility
            }
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                **model_kwargs
            )
            
            # Ensure embedding model is initialized
            if self.embedding_model is None:
                self.initialize_embedding_model()

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
                max_length=512,
                padding=True
            )

            # Move inputs to the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            try:
                with torch.inference_mode():
                    # Update generation parameters to avoid probability issues
                    outputs = self.llm_model.generate(
                        **inputs,
                        max_new_tokens=200,
                        num_beams=5,  # Add beam search
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=self.llm_tokenizer.eos_token_id,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        early_stopping=True,  # This is now valid with num_beams > 1
                        use_cache=True,
                        min_length=20,  # Add minimum length
                        bad_words_ids=[[self.llm_tokenizer.unk_token_id]],  # Avoid unknown tokens
                    )

                response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split("Answer: ")[-1].strip()

                if not response:
                    return "I apologize, but I couldn't generate a meaningful response about the products."

                return response

            except RuntimeError as e:
                logger.error(f"Runtime error in generate_response: {e}")
                # Fallback to simpler generation parameters
                try:
                    outputs = self.llm_model.generate(
                        **inputs,
                        max_new_tokens=100,
                        num_beams=1,  # No beam search
                        do_sample=False,  # Deterministic
                        temperature=1.0,
                        early_stopping=False,
                        use_cache=True
                    )
                    response = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.split("Answer: ")[-1].strip()
                    return response if response else "I apologize, but I had trouble generating a detailed response."
                except Exception as e2:
                    logger.error(f"Fallback generation failed: {e2}")
                    return "I apologize, but there was an error processing your query."

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I couldn't generate a response about the products."

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using the sentence-transformer model"""
        try:
            if self.embedding_model is None:
                logger.info("Embedding model not initialized, initializing now...")
                self.initialize_embedding_model()
                
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
        """Load model from shared storage or S3"""
        try:
            # Normalize model path
            model_path = AppConfig.get_model_s3_path(model_path)
            shared_path = AppConfig.get_model_path(model_path)
            logger.info(f"Loading model from shared path: {shared_path}")
            
            # Check if model is already in shared storage
            if os.path.exists(shared_path):
                logger.info(f"Found model in shared storage: {shared_path}")
                
                # Load embeddings and metadata
                embeddings_path = os.path.join(shared_path, "embeddings.npy")
                metadata_path = os.path.join(shared_path, "metadata.json")
                
                if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
                    logger.warning("Missing model files in shared storage, trying S3...")
                else:
                    embeddings = np.load(embeddings_path)
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    # Initialize models if needed
                    if not self.product_search.llm_model:
                        if not self.product_search.load_models(model_path):
                            raise ValueError("Failed to initialize models")
                    
                    return {
                        "embeddings": embeddings,
                        "metadata": metadata,
                        "loaded_at": datetime.now().isoformat()
                    }
            
            # If not in shared storage, download from S3
            logger.info("Downloading model files from S3...")
            if not AppConfig.ensure_model_files(model_path, self.s3_manager):
                raise ValueError(f"Failed to download model files for {model_path}")
            
            # Recursive call now that files are downloaded
            return self.load_model(model_path)
            
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
            logger.info(f"Starting search with model: {model_path}")
            
            # Load model and ensure all components are initialized
            model_data = self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            # Now load the models if not already loaded
            if not self.product_search.llm_model:
                self.product_search.load_models(model_path)

            # Generate query embedding
            query_embedding = self.product_search.generate_embedding(query)

            embeddings = model_data["embeddings"]

            # Calculate similarities
            similarities = np.dot(embeddings, query_embedding)
            top_k_idx = np.argsort(similarities)[-max_items:][::-1]

            # Load products data from shared volume first
            shared_path = AppConfig.get_model_path(model_path)
            products_csv_path = os.path.join(shared_path, "products.csv")
            
            if os.path.exists(products_csv_path):
                logger.info(f"Loading products from shared path: {products_csv_path}")
                products_df = pd.read_csv(products_csv_path)
            else:
                # Fallback to S3 only if not found locally
                logger.info("Products not found locally, trying S3...")
                products_df = self.s3_manager.get_csv_content(f"{model_path}/products.csv")
                
            if products_df is None:
                raise FileNotFoundError("Products data not found in shared volume or S3")

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

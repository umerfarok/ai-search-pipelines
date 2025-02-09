"""
RAG-based Search Service with Efficient Model Management
"""

import os
import logging
import threading
from datetime import datetime  # Fix import
from typing import Dict, List, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import boto3
from botocore.config import Config
import pandas as pd
import io
from collections import OrderedDict
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from config import AppConfig
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


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
        batch_size = 128

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
    """RAG-based search service with efficient model management"""

    def __init__(self):
        self.s3_manager = S3Manager()
        self.model_cache = ModelCache(AppConfig.MODEL_CACHE_SIZE)
        self.embedding_manager = EmbeddingManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_llm()
        self.vector_store = VectorStore()

    def _initialize_llm(self):
        """Initialize LLM for response generation"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "TheBloke/Mistral-7B-Instruct-v0.2-AWQ", trust_remote_code=True
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            self.llm = None
            self.tokenizer = None

    def _semantic_search(
        self, query: str, model_path: str, top_k: int = 5
    ) -> List[Dict]:
        """Perform semantic search using vector database"""
        try:
            # Extract config_id from model_path (e.g., "models/123456" -> "123456")
            config_id = model_path.split("/")[-1]
            collection_name = f"products_{config_id}"

            # Check if collection exists
            if not self.vector_store.collection_exists(collection_name):
                raise ValueError(f"No embeddings found for model: {model_path}")

            # Generate query embedding using the same model as training
            query_embedding = self.embedding_manager.generate_embedding(
                [query], model_name=self.get_model_name_from_metadata(config_id)
            )

            # Search in vector database
            results = self.vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding[0],
                limit=top_k,
            )

            # Format results with proper type handling
            return [self._format_search_result(result) for result in results]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _validate_embedding_model(
        self, collection_name: str, query_embedding: np.ndarray
    ) -> bool:
        """Validate embedding compatibility"""
        try:
            metadata = self.vector_store.get_collection_metadata(collection_name)
            if not metadata:
                logger.error(f"No metadata found for collection {collection_name}")
                return False

            expected_dim = metadata.get("embedding_dimension")
            if not expected_dim:
                logger.error("Embedding dimension not found in metadata")
                return False

            if query_embedding.shape[1] != expected_dim:
                logger.error(
                    f"Embedding dimension mismatch: expected {expected_dim}, got {query_embedding.shape[1]}"
                )
                return False

            return True
        except Exception as e:
            logger.error(f"Embedding validation failed: {e}")
            return False

    def get_model_name_from_metadata(self, config_id: str) -> str:
        try:
            metadata = self.vector_store.get_collection_metadata(
                f"products_{config_id}"
            )
            if metadata and "embeddingmodel" in metadata:
                return metadata["embeddingmodel"]
            return self.embedding_manager.default_model_name
        except Exception as e:
            logger.error(f"Failed to get model name from metadata: {e}")
            return self.embedding_manager.default_model_name

    def _validate_embeddings(self, embeddings: np.ndarray, model_name: str) -> bool:
        expected_dim = self.embedding_manager.get_model_dimension(model_name)
        if embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim}, got {embeddings.shape[1]}"
            )
        return True

    def _generate_response(self, query: str, products: List[Dict]) -> str:
        """Generate contextual response using LLM with improved prompt"""
        if not self.llm or not self.tokenizer:
            return "Response generation is not available."

        try:
            # Create detailed product context with pricing and features
            product_context = "\n".join(
                [
                    f"- Product: {p.get('name', 'Unknown')}\n"
                    f"  Description: {p.get('description', 'No description')}\n"
                    f"  Category: {p.get('category', 'Uncategorized')}\n"
                    f"  Price: {p.get('metadata', {}).get('price', 'Price not available')}\n"
                    f"  Features: {', '.join(str(v) for k, v in p.get('metadata', {}).items() if k not in ['price'])}"
                    for p in products
                ]
            )

            prompt = f"""<s>[INST] You are a helpful product search assistant. Based on these available products:

            {product_context}

            Help the user with this query: {query}

            Instructions:
            1. Focus only on products shown above
            2. If the query asks for pricing, mention specific prices
            3. Compare products if multiple options are suitable
            4. Highlight key features matching the query
            5. If no products match well, be honest about limitations
            6. If query is unclear, ask for clarification
            7. Keep response professional and concise

            Respond in this format:
            - Main recommendation(s)
            - Key features/pricing
            - Brief explanation why
            [/INST]"""

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            attention_mask = inputs["attention_mask"]
            outputs = self.llm.generate(
                inputs.input_ids,
                max_new_tokens=400,
                attention_mask=attention_mask,  # Increased for more detailed responses
                temperature=0.7,  # Slightly increased for more natural responses
                do_sample=True,
                top_p=0.95,  # Added for better quality
                repetition_penalty=1.2,  # Prevent repetitive text
                pad_token_id=self.tokenizer.eos_token_id,
            )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("[/INST]")[-1].strip()

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "Unable to generate response."

    def _format_search_result(self, result: Dict) -> Dict:
        """Format search result with proper ID handling"""
        try:
            print(result)
            payload = result.get("payload", {})
            return {
                "mongo_id": payload.get("mongo_id", ""),  # Original MongoDB ID
                "score": round(float(result.get("score", 0.0)), 4),
                "name": payload.get("name", ""),
                "description": payload.get("description", ""),
                "category": payload.get("category", ""),
                "metadata": payload.get("custom_metadata", {}),
                "qdrant_id": result.get("id", ""),  # Qdrant's UUID
            }
        except Exception as e:
            logger.error(f"Result formatting error: {str(e)}")
            return {"error": "Result formatting failed"}

    def search(self, query: str, model_path: str, top_k: int = 5) -> List[Dict]:
        """Search with collection verification and error handling"""
        try:
            # Config ID extraction with validation
            config_id = model_path.split("/")[-1].strip()
            if not config_id or len(config_id) != 24:
                raise ValueError("Invalid model path format")

            collection_name = f"products_{config_id}"

            # Collection existence check
            if not self.vector_store.collection_exists(collection_name):
                raise ValueError(f"Collection {collection_name} not found")

            # Metadata validation
            collection_metadata = self.vector_store.get_collection_metadata(
                collection_name
            )
            if not collection_metadata:
                raise ValueError("Collection metadata missing")

            # Model compatibility check
            model_name = collection_metadata.get("embedding_model")
            if not model_name:
                raise ValueError("Embedding model not specified in metadata")

            # Query processing
            query_embedding = self.embedding_manager.generate_embedding(
                [query], model_name
            )
            if query_embedding.shape[1] != collection_metadata["embedding_dimension"]:
                raise ValueError("Embedding dimension mismatch")

            # Search execution
            results = self.vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding[0].tolist(),
                limit=top_k,
            )

            return [self._format_search_result(r) for r in results]

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

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


search_service = SearchService()


@app.route("/search", methods=["POST"])
def search():
    """Enhanced search endpoint with RAG support"""
    try:
        data = request.get_json()
        if not data or "query" not in data or "model_path" not in data:
            return jsonify({"error": "Missing required fields"}), 400

        results = search_service.search(
            query=data["query"],
            model_path=data["model_path"],
            # max_items=data.get("max_items", 10),
        )
        llmResponse = search_service._generate_response(data["query"], results)
        print(llmResponse)

        return jsonify(llmResponse)

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=AppConfig.SERVICE_PORT, debug=False)

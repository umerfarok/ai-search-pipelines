"""
RAG-based Search Service with Efficient Model Management
"""

from concurrent.futures import ThreadPoolExecutor
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
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
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

    # Update these imports at the top of your file


from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch


class SearchService:
    def __init__(self):
        self.s3_manager = S3Manager()
        self.model_cache = ModelCache(AppConfig.MODEL_CACHE_SIZE)
        self.embedding_manager = EmbeddingManager()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_llm()
        self.vector_store = VectorStore()
        self._initialize_query_expansion()

    def _initialize_query_expansion(self):
        """Initialize lightweight model for query expansion"""
        try:
            self.query_expansion_model = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",
                device=self.device,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
            )
        except Exception as e:
            logger.warning(f"Query expansion model not loaded: {e}")

    def _expand_query(self, original_query: str) -> str:
        """Enhance search query with synonyms and related terms"""
        try:
            if self.query_expansion_model:
                expansion_prompt = f"Generate search synonyms for: {original_query}"
                expanded = self.query_expansion_model(
                    expansion_prompt,
                    max_length=50,
                    num_return_sequences=1,
                    do_sample=True,
                )[0]["generated_text"]
                return f"{original_query} {expanded}"
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
        return original_query

    def _initialize_llm(self):
        try:
            # Using a smaller, more efficient model
            model_name = "facebook/opt-350m"  # You can also try "facebook/opt-125m" for even smaller footprint

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                device=self.device,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                max_length=512,
            )
            logger.info("LLM initialized successfully using pipeline")
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

    def _format_frontend_response(
        self, results: List[Dict], generated_text: str
    ) -> Dict:
        """Structure response for frontend consumption"""
        return {
            "generated_response": generated_text,
            "results": [
                {
                    "product_id": res.get("mongo_id", ""),
                    "name": res.get("name", "Unknown"),
                    "price": res.get("metadata", {}).get("discount_price", "N/A"),
                    "image_url": res.get("metadata", {}).get("image_url", ""),
                    "score": res.get("score", 0),
                    "category": res.get("category", ""),
                }
                for res in results
            ],
            "search_metadata": {
                "result_count": len(results),
                "timestamp": datetime.now().isoformat(),
            },
        }

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
        """Generate contextual response using LLM with corrected data access"""
        if not self.llm:
            return "Response generation is not available."

        try:
            # Create detailed product context with direct access to product fields
            # The search results now contain fields at the top level rather than nested in metadata
            product_context = "\n".join(
                [
                    f"- Product: {p.get('name', 'Unknown')}\n"
                    f"  Description: {p.get('description', 'No description')}\n"
                    f"  Category: {p.get('category', 'Uncategorized')}\n"
                    f"  Price: {p.get('metadata', {}).get('discount_price', 'Price not available')}\n"
                    f"  Ratings: {p.get('metadata', {}).get('no_of_ratings', 'No ratings')}\n"
                    f"  Product ID: {p.get('qdrant_id', 'No ID')}"
                    for p in products
                ]
            )

            prompt = f"""As a product recommendation assistant, analyze these products:

            {product_context}

            User Query: {query}

            Please provide a helpful response that:
            1. Recommends the most relevant products
            2. Mentions specific prices and ratings
            3. Compares features if multiple products are suitable
            4. Explains why these products match the query
            5. Keeps the response concise and clear

            Response:"""

            # Generate response using pipeline
            response = self.llm(
                prompt,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.2,
                num_return_sequences=1,
            )[0]["generated_text"]

            # Extract only the generated part after the prompt
            generated_response = response.split("Response:")[-1].strip()

            # If the response is empty or too short, provide a fallback
            if len(generated_response) < 50:
                return self._generate_fallback_response(products)

            return generated_response

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "Unable to generate response."

    def _generate_fallback_response(self, products: List[Dict]) -> str:
        """Generate a simple structured response when LLM fails, with corrected data access"""
        try:
            relevant_products = []
            for p in products:
                relevant_products.append(
                    {
                        "name": p.get("name", "Unknown"),
                        "price": p.get("metadata", {}).get(
                            "discount_price", "Price not available"
                        ),
                        "ratings": p.get("metadata", {}).get(
                            "no_of_ratings", "No ratings"
                        ),
                    }
                )

            response = "Here are some relevant products:\n\n"
            for p in relevant_products:
                response += f"- {p['name']}\n"
                response += f"  Price: {p['price']}\n"
                response += f"  Ratings: {p['ratings']}\n\n"

            return response

        except Exception as e:
            logger.error(f"Fallback response generation failed: {e}")
            return "Unable to generate product recommendations."

    def _format_search_result(self, result: Dict) -> Dict:
        """Format search result with corrected field mapping"""
        try:
            metadata = result.get("metadata", {})
            custom_metadata = metadata.get("custom_metadata", {})

            return {
                "mongo_id": metadata.get("mongo_id", ""),
                "score": round(float(result.get("score", 0.0)), 4),
                "name": metadata.get("name", ""),
                "description": metadata.get("description", ""),
                "category": metadata.get("category", ""),
                "metadata": custom_metadata,  # Contains discount_price and no_of_ratings
                "qdrant_id": result.get("id", ""),
            }
        except Exception as e:
            logger.error(f"Result formatting error: {str(e)}")
            return {"error": "Result formatting failed"}

    def search(
        self, query: str, model_path: str, top_k: int = 5, filters: Dict = None
    ) -> List[Dict]:
        """Enhanced search with query expansion and filters"""
        try:
            # Query processing improvements
            query = self._preprocess_query(query)
            expanded_query = self._expand_query(query)

            # Check cache first
            cache_key = f"{expanded_query}:{model_path}:{top_k}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Enhanced search with filters
            results = self._semantic_search(expanded_query, model_path, top_k)

            # Apply post-filters
            print(results)
            if filters:
                results = self._apply_filters(results, filters)

            # Cache results
            self.cache[cache_key] = results
            return results

        except Exception as e:
            logger.error(f"Enhanced search failed: {str(e)}")
            return []

    def _preprocess_query(self, query: str) -> str:
        """Clean and normalize search query"""
        query = query.lower().strip()
        query = re.sub(r"[^\w\s-]", "", query)  # Remove special chars
        return query

    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply post-search filters"""
        filtered = []
        for res in results:
            metadata = res.get("metadata", {})
            match = True

            # Price filtering
            if "max_price" in filters:
                price = float(metadata.get("discount_price", 0))
                if price > filters["max_price"]:
                    match = False

            # Category filtering
            if "categories" in filters:
                if res.get("category", "").lower() not in [
                    c.lower() for c in filters["categories"]
                ]:
                    match = False

            if match:
                filtered.append(res)
        return filtered

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
    """Enhanced endpoint with frontend support"""
    try:
        data = request.get_json()
        # Validate input
        if not data or "query" not in data or "model_path" not in data:
            return jsonify({"error": "Missing required fields"}), 400

        # Process with timeout
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                search_service.search,
                query=data["query"],
                model_path=data["model_path"],
                top_k=data.get("top_k", 10),
                filters=data.get("filters", {}),
            )
            results = future.result(timeout=5)  # 5 second timeout

    
        llm_response = search_service._generate_response(data["query"], results)
        print("llmresponse", llm_response)
        response = search_service._format_frontend_response(results, llm_response)
        print(response)

        return jsonify(response)

    except TimeoutError:
        logger.warning("Search request timed out")
        return jsonify({"error": "Request timeout"}), 504
    except Exception as e:
        logger.error(f"Endpoint error: {str(e)}")
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

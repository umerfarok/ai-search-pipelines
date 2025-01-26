import os
import logging
from flask import Flask, request, jsonify
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import redis
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from typing import Dict, List, Optional, Union
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class AppConfig:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    MODEL_CACHE_SIZE = 2
    MIN_SCORE = 0.2
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_REGION = os.getenv("AWS_REGION", "us-east-1")
    AWS_SSL_VERIFY = os.getenv("AWS_SSL_VERIFY", "true")
    AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")


class S3Manager:
    def __init__(self):
        # Configure boto3 client
        config = Config(retries=dict(max_attempts=3))

        # Get endpoint URL for LocalStack if set
        endpoint_url = AppConfig.AWS_ENDPOINT_URL
        verify_ssl = AppConfig.AWS_SSL_VERIFY.lower() == "true"

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AppConfig.AWS_ACCESS_KEY,
            aws_secret_access_key=AppConfig.AWS_SECRET_KEY,
            region_name=AppConfig.S3_REGION,
            endpoint_url=endpoint_url,  # Will be None in production
            verify=verify_ssl,  # False for LocalStack
            config=config,
        )
        self.bucket = AppConfig.S3_BUCKET
        logger.info(f"Initialized S3Manager with bucket: {self.bucket}")
        if endpoint_url:
            logger.info(f"Using custom endpoint: {endpoint_url}")

    def download_file(self, s3_path: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3.download_file(self.bucket, s3_path, local_path)
            logger.info(
                f"Successfully downloaded s3://{self.bucket}/{s3_path} to {local_path}"
            )
            return True
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            return False


class ModelManager:
    def __init__(self):
        self.model_cache = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.s3_manager = S3Manager()
        logger.info(f"Using device: {self.device}")

    def load_model(self, model_path: str) -> Optional[Dict]:
        try:
            model_path = self._resolve_path(model_path)

            if model_path in self.model_cache:
                return self.model_cache[model_path]

            logger.info(f"Loading model from {model_path}")

            # Download model files from S3
            local_model_dir = f"/tmp/models/{Path(model_path).name}"
            os.makedirs(local_model_dir, exist_ok=True)

            required_files = ["metadata.json", "model.pt", "embeddings.npy"]
            for file in required_files:
                s3_path = f"{model_path}/{file}"
                local_path = f"{local_model_dir}/{file}"
                if not self.s3_manager.download_file(s3_path, local_path):
                    raise FileNotFoundError(f"Failed to download {file} from S3")

            # Load metadata
            with open(f"{local_model_dir}/metadata.json", "r") as f:
                metadata = json.load(f)

            # Initialize model
            model = SentenceTransformer(metadata["model_name"])
            model.to(self.device)

            # Load saved weights
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
                "last_used": (
                    torch.cuda.current_stream().record_event()
                    if torch.cuda.is_available()
                    else None
                ),
            }

            # Manage cache size
            if len(self.model_cache) >= AppConfig.MODEL_CACHE_SIZE:
                oldest_path = min(
                    self.model_cache.keys(),
                    key=lambda k: id(self.model_cache[k]["last_used"]),
                )
                del self.model_cache[oldest_path]

            self.model_cache[model_path] = model_data
            return model_data

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def _resolve_path(self, path: str) -> str:
        return path.strip("/")


class SearchProcessor:
    def __init__(self):
        self.model_manager = ModelManager()
        self.s3_manager = S3Manager()

    def search(self, model_path: str, query: str, max_items: int = 5) -> Dict:
        try:
            # Load model
            model_data = self.model_manager.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            model = model_data["model"]
            embeddings = model_data["embeddings"]
            metadata = model_data["metadata"]

            # Generate query embedding
            query_embedding = (
                model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
                .cpu()
                .numpy()
            )

            # Calculate similarities
            similarities = np.dot(query_embedding, embeddings.T)
            top_k_idx = np.argsort(similarities)[-max_items:][::-1]

            # Load product data from S3
            local_products_path = f"/tmp/products/{Path(model_path).name}_products.csv"
            s3_products_path = f"{model_path}/products.csv"

            if not self.s3_manager.download_file(s3_products_path, local_products_path):
                raise FileNotFoundError("Products file not found in S3")

            # Read CSV with proper error handling
            try:
                products_df = pd.read_csv(local_products_path)
            except Exception as e:
                logger.error(f"Error reading products CSV: {e}")
                raise

            # Prepare results
            results = []
            schema = metadata["config"]["schema_mapping"]

            for idx in top_k_idx:
                score = float(similarities[idx])
                if score < AppConfig.MIN_SCORE:
                    continue

                try:
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
                    }

                    # Add metadata from custom columns
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
                    "model_version": metadata.get("version", "unknown"),
                    "schema_mapping": schema,
                },
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise


# Initialize search processor
search_processor = SearchProcessor()


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        if not data or "model_path" not in data or "query" not in data:
            return jsonify({"error": "Invalid request"}), 400

        model_path = data["model_path"]
        query = data["query"]
        max_items = data.get("max_items", 5)

        logger.info(f"Processing search request for model: {model_path}")
        logger.info(f"Query: {query}")

        results = search_processor.search(model_path, query, max_items)
        return jsonify(results)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Search error: {error_msg}")
        return jsonify({"error": error_msg}), 500


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 5000))
    app.run(host="0.0.0.0", port=port)

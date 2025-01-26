import os
import logging
from flask import Flask, request, jsonify
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
import redis
from typing import Dict, List, Optional
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import pandas as pd
import io
from dataclasses import dataclass
from datetime import datetime
import threading
from collections import OrderedDict

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
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3.download_file(self.bucket, s3_path, local_path)
            logger.info(f"Downloaded s3://{self.bucket}/{s3_path} to {local_path}")
            return True
        except ClientError as e:
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


class SearchService:
    def __init__(self):
        self.s3_manager = S3Manager()
        self.model_cache = ModelCache(AppConfig.MODEL_CACHE_SIZE)
        self.device = AppConfig.DEVICE
        logger.info(f"Initialized SearchService using device: {self.device}")

    def load_model(self, model_path: str) -> Optional[Dict]:
        """Load model and its artifacts from S3"""
        try:
            # Check cache first
            cached_model = self.model_cache.get(model_path)
            if cached_model:
                logger.info(f"Using cached model for {model_path}")
                return cached_model

            logger.info(f"Loading model from {model_path}")

            # Create temporary directory for model files
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

            # Initialize model
            model = SentenceTransformer(metadata["model_name"])
            model.to(self.device)

            # Load model state
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

    def search(
        self,
        query: str,
        model_path: str,
        max_items: int = 10,
        filters: Optional[Dict] = None,
    ) -> Dict:
        """Perform semantic search using the specified model"""
        try:
            # Load model
            model_data = self.load_model(model_path)
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
            similarities = np.dot(embeddings, query_embedding)
            top_k_idx = np.argsort(similarities)[-max_items:][::-1]

            # Load products data
            products_path = f"{model_path}/products.csv"
            products_df = self.s3_manager.get_csv_content(products_path)
            if products_df is None:
                raise FileNotFoundError("Products data not found")

            # Apply filters if provided
            if filters:
                products_df = self.apply_filters(
                    products_df, filters, metadata["config"]["schema_mapping"]
                )

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
                    "model_path": model_path,
                    "schema": schema,
                },
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    def apply_filters(
        self, df: pd.DataFrame, filters: Dict, schema_mapping: Dict
    ) -> pd.DataFrame:
        """Apply filters to the DataFrame"""
        filtered_df = df.copy()

        try:
            for field, value in filters.items():
                # Handle standard columns
                if field in ["category", "name", "description"]:
                    column = schema_mapping.get(f"{field}_column")
                    if column and column in filtered_df.columns:
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
                                if (
                                    isinstance(value, list) and len(value) == 2
                                ):  # Range filter
                                    filtered_df = filtered_df[
                                        (filtered_df[column] >= value[0])
                                        & (filtered_df[column] <= value[1])
                                    ]
                                else:  # Exact match or contains
                                    filtered_df = filtered_df[
                                        filtered_df[column]
                                        .astype(str)
                                        .str.contains(str(value), case=False, na=False)
                                    ]

        except Exception as e:
            logger.error(f"Error applying filters: {e}")

        return filtered_df


# Initialize search service
search_service = SearchService()


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

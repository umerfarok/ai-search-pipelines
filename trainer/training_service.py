# training_service.py

import os
import logging
import time
from typing import Dict, List, Optional
import torch
import numpy as np
import json
from pathlib import Path
import redis
from datetime import datetime
import threading
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from flask import Flask, request, jsonify
import transformers
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import pandas as pd
import io
from enum import Enum
from config import AppConfig
from model_initializer import ModelInitializer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
transformers.utils.move_cache()
app = Flask(__name__)


class ModelStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class EmbeddingManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_models = {}
        self.model_load_lock = threading.Lock()
        self.max_seq_length = 512

        # Ensure cache directories exist
        AppConfig.setup_cache_dirs()

        # Set up HuggingFace authentication if token is available
        self.hf_token = AppConfig.HF_TOKEN
        if self.hf_token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.hf_token

    def _get_model_info(self, model_key: str) -> Optional[Dict]:
        """Get model information handling both keys and full paths."""
        # First check if it's a known model key
        if model_key in AppConfig.MODEL_MAPPINGS:
            return {
                "key": model_key,
                "path": AppConfig.MODEL_MAPPINGS[model_key]["path"],
                **AppConfig.MODEL_MAPPINGS[model_key],
            }

        # Then check if it's a full path that matches any known models
        for key, info in AppConfig.MODEL_MAPPINGS.items():
            if info["path"] == model_key:
                return {"key": key, "path": model_key, **info}

        # If not found, try to use as direct path with fallback
        if model_key != AppConfig.DEFAULT_MODEL:
            logger.warning(
                f"Model {model_key} not found in mappings. Falling back to default."
            )
            return self._get_model_info(AppConfig.DEFAULT_MODEL)

        # Last resort - use as direct path
        return {
            "key": model_key,
            "path": model_key,
            "name": model_key.split("/")[-1],
            "description": "Custom model",
            "tags": ["custom"],
            "is_default": False,
        }

    def get_model(self, model_key: str = None):
        """Load a model with improved error handling and fallback."""
        if not model_key:
            model_key = AppConfig.DEFAULT_MODEL

        model_info = self._get_model_info(model_key)
        if not model_info:
            raise ValueError(f"Unsupported model: {model_key}")

        model_path = model_info["path"]
        if model_path in self.embedding_models:
            return self.embedding_models[model_path]

        try:
            with self.model_load_lock:
                logger.info(f"Loading model: {model_path} (key: {model_key})")

                # Try loading the model with explicit cache configuration
                model = SentenceTransformer(
                    model_path,
                    cache_folder=AppConfig.TRANSFORMER_CACHE,
                    device=self.device,
                    use_auth_token=self.hf_token,
                )

                # Set sequence length
                if hasattr(model, "max_seq_length"):
                    model.max_seq_length = self.max_seq_length
                else:
                    model._first_module().max_seq_length = self.max_seq_length

                self.embedding_models[model_path] = model
                logger.info(f"Successfully loaded model: {model_path}")
                return model

        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            if model_key != AppConfig.DEFAULT_MODEL:
                logger.info(f"Falling back to default model: {AppConfig.DEFAULT_MODEL}")
                return self.get_model(AppConfig.DEFAULT_MODEL)
            raise RuntimeError(
                f"Model loading failed and fallback failed: {model_path}"
            ) from e

    def generate_embedding(self, text: List[str], model_key: str = None) -> np.ndarray:
        """Generate embeddings with improved batch handling and error recovery."""
        model = self.get_model(model_key)
        if model is None:
            raise ValueError(f"Model {model_key} not initialized")

        # Adjust batch size based on available memory
        batch_size = 32 if torch.cuda.is_available() else 16

        try:
            with torch.no_grad():
                embeddings = model.encode(
                    text,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    device=self.device,
                )
                return embeddings.cpu().numpy()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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


class ProductTrainer:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.s3_manager = S3Manager()

    def _process_text(self, row: pd.Series, schema: Dict, config: Dict) -> str:
        text_parts = []
        field_mapping = {
            "name": schema.get("name_column"),
            "description": schema.get("description_column"),
            "category": schema.get("category_column"),
        }

        for field, column in field_mapping.items():
            if column and column in row:
                text_parts.append(str(row[column]))

        base_text = " ".join(filter(None, text_parts))

        # Add custom column handling
        if "custom_columns" in schema:
            for col in schema["custom_columns"]:
                if col["name"] in row and col.get("role") == "training":
                    text_parts.append(str(row[col["name"]]))

        return " ".join(filter(None, text_parts))

    def _load_data(self, config: Dict) -> Optional[pd.DataFrame]:
        try:
            data_source = config["data_source"]
            current_file = data_source["location"]

            df = self.s3_manager.get_csv_content(current_file)
            if df is None:
                raise ValueError(f"Failed to load CSV from {current_file}")

            if config.get("mode") == "append" and config.get("previous_version"):
                prev_path = f"models/{config['previous_version']}/products.csv"
                prev_df = self.s3_manager.get_csv_content(prev_path)
                if prev_df is not None:
                    df = pd.concat([prev_df, df], ignore_index=True)

            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def train(self, config: Dict) -> bool:
        try:
            config_id = config.get("_id")
            model_name = config["training_config"]["embedding_model"]
            logger.info(f"Using embedding model: {model_name}")

            df = self._load_data(config)
            if df is None or len(df) == 0:
                raise ValueError("No valid training data found")

            processed_texts = [
                self._process_text(row, config["schema_mapping"], config)
                for _, row in df.iterrows()
            ]

            embeddings = self.embedding_manager.generate_embedding(
                processed_texts, model_name
            )

            model_path = f"models/{config_id}"
            model_dir = os.path.join(AppConfig.BASE_MODEL_DIR, model_path)

            if not self._save_model_files(
                model_dir, embeddings, df, config, processed_texts
            ):
                raise Exception("Failed to save model files")

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

            np.save(os.path.join(model_dir, "embeddings.npy"), embeddings)
            df.to_csv(os.path.join(model_dir, "products.csv"), index=False)

            with open(os.path.join(model_dir, "processed_texts.json"), "w") as f:
                json.dump(processed_texts, f)

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "num_samples": len(df),
                "embedding_shape": embeddings.shape,
                "models": {
                    "embeddings": config["training_config"]["embedding_model"],
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


class TrainingWorker:
    def __init__(self):
        self.trainer = ProductTrainer()
        self.redis = redis.Redis(
            host=AppConfig.REDIS_HOST,
            port=AppConfig.REDIS_PORT,
            password=AppConfig.REDIS_PASSWORD,
            decode_responses=True,
        )
        self.should_stop = False
        self.current_job = None
        self._worker_thread = None

    def process_job(self, job: Dict):
        try:
            config = job.get("config", {})
            config_id = config.get("_id")
            print(job)
            if not config_id:
                raise ValueError("Invalid job structure - missing config ID")

            logger.info(f"Processing job for config: {config_id}")
            self._update_status(config_id, ModelStatus.PROCESSING, progress=0)

            # Convert schema mapping keys if needed
            if "schema_mapping" in config:
                config["schema_mapping"] = {
                    "idcolumn": config["schema_mapping"].get("id_column"),
                    "namecolumn": config["schema_mapping"].get("name_column"),
                    "descriptioncolumn": config["schema_mapping"].get(
                        "description_column"
                    ),
                    "categorycolumn": config["schema_mapping"].get("category_column"),
                    "customcolumns": config["schema_mapping"].get("custom_columns", []),
                }

            success = self.trainer.train(config)

            if success:
                self._update_status(config_id, ModelStatus.COMPLETED, progress=100)
            else:
                self._update_status(
                    config_id, ModelStatus.FAILED, error="Training failed"
                )

        except Exception as e:
            logger.error(f"Error processing job: {e}")
            if "config" in job and "_id" in job["config"]:
                self._update_status(
                    job["config"]["_id"], ModelStatus.FAILED, error=str(e)
                )

    def _update_status(
        self,
        config_id: str,
        status: ModelStatus,
        progress: float = None,
        error: str = None,
    ):
        try:
            status_data = {
                "status": status.value,
                "timestamp": datetime.now().isoformat(),
            }

            if progress is not None:
                status_data["training_stats"] = {
                    "progress": progress,
                    "start_time": datetime.now().isoformat() if progress == 0 else None,
                    "end_time": datetime.now().isoformat() if progress == 100 else None,
                }
            if error:
                status_data["error"] = error

            key = f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}"
            self.redis.set(key, json.dumps(status_data), ex=86400)

        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def _worker_loop(self):
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
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)

    def start(self):
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.should_stop = False
            self._worker_thread = threading.Thread(target=self._worker_loop)
            self._worker_thread.daemon = True
            self._worker_thread.start()
            logger.info("Training worker started")

    def stop(self):
        logger.info("Stopping training worker")
        self.should_stop = True
        if self._worker_thread:
            self._worker_thread.join(timeout=5)


# Initialize services
worker = TrainingWorker()


# Flask routes
@app.route("/train", methods=["POST"])
def train():
    try:
        data = request.get_json()
        if not data or "config" not in data:
            return jsonify({"error": "Invalid request"}), 400

        # Add job to training queue
        job_data = json.dumps(data)
        worker.redis.lpush(AppConfig.TRAINING_QUEUE, job_data)
        return jsonify({"status": "queued", "config_id": data["config"]["_id"]})

    except Exception as e:
        logger.error(f"Error submitting training job: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/status/<config_id>")
def get_status(config_id):
    try:
        status_key = f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}"
        status = worker.redis.get(status_key)

        if status:
            return jsonify(json.loads(status))

        if worker.current_job and worker.current_job["config"]["_id"] == config_id:
            return jsonify(
                {
                    "status": ModelStatus.PROCESSING.value,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return jsonify({"error": "Status not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/models")
def get_available_models():
    """Get available embedding models info"""
    return jsonify(
        {
            "models": EmbeddingManager().MODEL_MAPPINGS,
            "default": AppConfig.DEFAULT_MODEL,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )


@app.route("/control/start", methods=["POST"])
def start_worker():
    try:
        worker.start()
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/control/stop", methods=["POST"])
def stop_worker():
    try:
        worker.stop()
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def initialize_service():
    """Initialize service with improved error handling and setup."""
    logger.info("Initializing service...")

    try:
        AppConfig.setup_cache_dirs()
        worker.start()
        if hasattr(AppConfig, "REQUIRED_MODELS") and AppConfig.REQUIRED_MODELS:
            embedding_manager = EmbeddingManager()
            for model_name in AppConfig.REQUIRED_MODELS:
                try:
                    logger.info(f"Pre-loading model: {model_name}")
                    embedding_manager.get_model(model_name)
                except Exception as e:
                    logger.error(f"Failed to pre-load model {model_name}: {e}")

        logger.info("Service initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False


if __name__ == "__main__":
    if not initialize_service():
        logger.error("Service initialization failed")

    app.run(
        host="0.0.0.0",
        port=AppConfig.SERVICE_PORT,
        use_reloader=False,  # Disable reloader to prevent duplicate model loading
    )

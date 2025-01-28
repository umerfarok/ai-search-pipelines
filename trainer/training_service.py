import os
import logging
from flask import Flask, request, jsonify
import torch
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
from pathlib import Path
import redis
from datetime import datetime
import time
import threading
from typing import Dict, List, Optional, Union, Tuple
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import requests
import io
from enum import Enum
from dataclasses import dataclass
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


class ModelStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class TrainingConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"  # Upgraded model
    batch_size: int = 32
    epochs: int = 3
    max_seq_length: int = 256
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    validation_split: float = 0.1


class AppConfig:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    TRAINING_QUEUE = os.getenv("TRAINING_QUEUE", "training_queue")
    MODEL_STATUS_PREFIX = os.getenv("MODEL_PREFIX", "model_status:")
    SERVICE_PORT = int(os.getenv("SERVICE_PORT", 5001))
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_REGION = os.getenv("AWS_REGION", "us-east-1")
    API_HOST = os.getenv("API_HOST", "http://api:8080")
    AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        # Lowercase the text
        text = text.lower()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and token.isalnum()
        ]

        return " ".join(tokens)

    def create_training_text(self, row: pd.Series, schema: Dict) -> str:
        """Create enriched training text from a row using schema mapping"""
        text_parts = []

        # Process main columns
        if schema.get("name_column"):
            text_parts.append(f"name: {row.get(schema['name_column'], '')}")

        if schema.get("description_column"):
            text_parts.append(
                f"description: {row.get(schema['description_column'], '')}"
            )

        if schema.get("category_column"):
            text_parts.append(f"category: {row.get(schema['category_column'], '')}")

        # Process custom training columns
        for col in schema.get("custom_columns", []):
            if col["role"] == "training":
                text_parts.append(
                    f"{col['standard_column']}: {row.get(col['user_column'], '')}"
                )

        # Join and preprocess
        text = " | ".join(filter(None, text_parts))
        return self.preprocess(text)


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

    def get_csv_content(self, s3_path: str) -> Optional[pd.DataFrame]:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_path)
            return pd.read_csv(io.BytesIO(response["Body"].read()))
        except Exception as e:
            logger.error(f"Failed to read CSV from S3: {e}")
            return None

    def upload_file(self, local_path: str, s3_path: str) -> bool:
        try:
            extra_args = {
                "ContentType": {
                    ".json": "application/json",
                    ".pt": "application/octet-stream",
                    ".npy": "application/octet-stream",
                    ".csv": "text/csv",
                }.get(Path(local_path).suffix, "application/octet-stream")
            }

            self.s3.upload_file(local_path, self.bucket, s3_path, ExtraArgs=extra_args)
            return True
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False


class ModelTrainer:
    def __init__(self):
        self.device = AppConfig.DEVICE
        self.preprocessor = TextPreprocessor()
        self.s3_manager = S3Manager()
        self.training_config = TrainingConfig()

    def prepare_training_data(
        self, df: pd.DataFrame, config: Dict
    ) -> Tuple[List[str], np.ndarray]:
        """Prepare training data with enhanced text processing"""
        texts = []
        schema = config["schema_mapping"]

        for _, row in df.iterrows():
            processed_text = self.preprocessor.create_training_text(row, schema)
            texts.append(processed_text)

        return texts

    def train_model(
        self, texts: List[str], config_id: str, callback: Optional[callable] = None
    ) -> Tuple[SentenceTransformer, np.ndarray]:
        """Train the model with progress tracking"""
        # Initialize model
        model = SentenceTransformer(self.training_config.model_name)
        model.to(self.device)

        # Prepare training data
        train_dataloader = DataLoader(
            texts, batch_size=self.training_config.batch_size, shuffle=True
        )

        # Training loop with progress tracking
        model.fit(
            train_objectives=[
                (train_dataloader, losses.MultipleNegativesRankingLoss(model))
            ],
            epochs=self.training_config.epochs,
            warmup_steps=self.training_config.warmup_steps,
            callback=callback,
        )

        # Generate embeddings
        embeddings = model.encode(
            texts,
            batch_size=self.training_config.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        return model, embeddings.cpu().numpy()

    def save_artifacts(
        self,
        model: SentenceTransformer,
        embeddings: np.ndarray,
        texts: List[str],
        df: pd.DataFrame,
        config: Dict,
        temp_dir: str,
    ) -> bool:
        """Save all training artifacts"""
        try:
            os.makedirs(temp_dir, exist_ok=True)

            # Save model state
            model_path = f"{temp_dir}/model.pt"
            torch.save(model.state_dict(), model_path)

            # Save embeddings
            embeddings_path = f"{temp_dir}/embeddings.npy"
            np.save(embeddings_path, embeddings)

            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "model_info": {
                    "name": self.training_config.model_name,
                    "params": self.training_config.__dict__,
                },
                "data_info": {
                    "num_samples": len(texts),
                    "embedding_dim": embeddings.shape[1],
                    "columns": df.columns.tolist(),
                },
            }

            metadata_path = f"{temp_dir}/metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Save processed data
            data_path = f"{temp_dir}/data.csv"
            df.to_csv(data_path, index=False)

            # Upload to S3
            base_path = config["model_path"]
            files = {
                "model.pt": model_path,
                "embeddings.npy": embeddings_path,
                "metadata.json": metadata_path,
                "data.csv": data_path,
            }

            for name, path in files.items():
                if not self.s3_manager.upload_file(path, f"{base_path}/{name}"):
                    raise Exception(f"Failed to upload {name}")

            return True

        except Exception as e:
            logger.error(f"Error saving artifacts: {e}")
            return False


class TrainingWorker:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.redis = redis.Redis(
            host=AppConfig.REDIS_HOST,
            port=AppConfig.REDIS_PORT,
            password=AppConfig.REDIS_PASSWORD,
            decode_responses=True,
        )
        self.should_stop = False
        self.current_job = None

    def update_status(
        self, config_id: str, status: str, progress: float = None, error: str = None
    ):
        """Update training status"""
        try:
            # Update Redis
            status_data = {"status": status, "timestamp": datetime.now().isoformat()}
            if progress is not None:
                status_data["progress"] = progress
            if error:
                status_data["error"] = error

            self.redis.set(
                f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}",
                json.dumps(status_data),
                ex=86400,  # 24 hour expiry
            )

            # Update API
            requests.put(
                f"{AppConfig.API_HOST}/config/status/{config_id}",
                json=status_data,
                timeout=5,
            )

        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def process_job(self, job: Dict):
        """Process a training job"""
        config = job["config"]
        config_id = job["config_id"]

        try:
            self.update_status(config_id, ModelStatus.PROCESSING.value, progress=0)

            # Load and preprocess data
            df = self.trainer.s3_manager.get_csv_content(
                config["data_source"]["location"]
            )
            if df is None:
                raise Exception("Failed to load training data")

            # Prepare training data
            texts = self.trainer.prepare_training_data(df, config)
            self.update_status(config_id, ModelStatus.PROCESSING.value, progress=20)

            # Train model
            def progress_callback(progress: float):
                self.update_status(
                    config_id,
                    ModelStatus.PROCESSING.value,
                    progress=20 + (progress * 60),
                )

            model, embeddings = self.trainer.train_model(
                texts, config_id, progress_callback
            )

            # Save artifacts
            temp_dir = f"/tmp/training/{config['model_path']}"
            if self.trainer.save_artifacts(
                model, embeddings, texts, df, config, temp_dir
            ):
                self.update_status(config_id, ModelStatus.COMPLETED.value, progress=100)
            else:
                raise Exception("Failed to save artifacts")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self.update_status(config_id, ModelStatus.FAILED.value, error=str(e))

    def start(self):
        """Start the training worker"""
        self.should_stop = False
        thread = threading.Thread(target=self._worker_loop)
        thread.daemon = True
        thread.start()
        logger.info("Training worker started")

    def _worker_loop(self):
        """Main worker loop"""
        while not self.should_stop:
            try:
                # Get next job from queue
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

    def stop(self):
        """Stop the training worker"""
        self.should_stop = True
        logger.info("Training worker stopped")


# Initialize worker
worker = TrainingWorker()


@app.route("/health")
def health():
    """Health check endpoint"""
    try:
        redis_ok = worker.redis.ping()
        return jsonify(
            {
                "status": "healthy",
                "redis": redis_ok,
                "worker": {
                    "running": not worker.should_stop,
                    "current_job": (
                        worker.current_job["config_id"] if worker.current_job else None
                    ),
                },
            }
        )
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/status/<config_id>")
def get_status(config_id):
    """Get training status for a specific configuration"""
    try:
        # Check Redis for status
        status_key = f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}"
        status = worker.redis.get(status_key)

        if status:
            return jsonify(json.loads(status))

        # Check current job
        if worker.current_job and worker.current_job["config_id"] == config_id:
            return jsonify(
                {
                    "status": ModelStatus.PROCESSING.value,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return jsonify({"status": "not_found"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/queue")
def get_queue_status():
    """Get current queue status"""
    try:
        queue_length = worker.redis.llen(AppConfig.TRAINING_QUEUE)

        # Get queued jobs
        jobs = []
        if queue_length > 0:
            job_data = worker.redis.lrange(AppConfig.TRAINING_QUEUE, 0, -1)
            for job in job_data:
                try:
                    parsed_job = json.loads(job)
                    jobs.append(
                        {
                            "config_id": parsed_job["config_id"],
                            "timestamp": parsed_job.get("timestamp", "unknown"),
                        }
                    )
                except json.JSONDecodeError:
                    continue

        return jsonify(
            {
                "queue_length": queue_length,
                "current_job": (
                    worker.current_job["config_id"] if worker.current_job else None
                ),
                "queued_jobs": jobs,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/control/start", methods=["POST"])
def start_worker():
    """Start the training worker"""
    try:
        worker.start()
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/control/stop", methods=["POST"])
def stop_worker():
    """Stop the training worker"""
    try:
        worker.stop()
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def cleanup():
    """Cleanup handler for graceful shutdown"""
    logger.info("Shutting down training service...")
    worker.stop()
    logger.info("Training service shutdown complete")
    
    
if __name__ == "__main__":
    # Register cleanup handler
    import atexit
    atexit.register(cleanup)

    # Start worker
    worker.start()

    # Run Flask app
    port = int(os.getenv("SERVICE_PORT", 5001))  # Note: Training service uses port 5001 by default
    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False  # Disable reloader to prevent multiple worker instances
    )
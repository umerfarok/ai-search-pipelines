import os
import logging
from flask import Flask, request, jsonify
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
from pathlib import Path
import redis
from datetime import datetime
import time
import threading
from typing import Dict, List, Optional, Union
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import requests
import io
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ModelStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class APPConfig:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    TRAINING_QUEUE = os.getenv("TRAINING_QUEUE", "training_queue")
    MODEL_PREFIX = os.getenv("MODEL_PREFIX", "model_status:")
    SERVICE_PORT = int(os.getenv("SERVICE_PORT", 5001))
    MODEL_NAME = "all-mpnet-base-v2"
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_REGION = os.getenv("AWS_REGION", "us-east-1")
    API_HOST = os.getenv("API_HOST", "http://localhost:8080")
    AWS_SSL_VERIFY = os.getenv("AWS_SSL_VERIFY", "true")
    AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")

class S3Manager:
    def __init__(self):
        config = Config(
            retries=dict(max_attempts=3),
            s3={'addressing_style': 'path'}
        )

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=APPConfig.AWS_ACCESS_KEY,
            aws_secret_access_key=APPConfig.AWS_SECRET_KEY,
            region_name=APPConfig.S3_REGION,
            endpoint_url=APPConfig.AWS_ENDPOINT_URL,
            verify=APPConfig.AWS_SSL_VERIFY.lower() == "true",
            config=config,
        )
        self.bucket = APPConfig.S3_BUCKET
        logger.info(f"Initialized S3Manager with bucket: {self.bucket}")

    def get_csv_content(self, s3_path: str) -> Optional[pd.DataFrame]:
        try:
            if s3_path.startswith('s3://'):
                s3_path = '/'.join(s3_path.split('/')[3:])
            
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_path)
            return pd.read_csv(io.BytesIO(response["Body"].read()))
        except ClientError as e:
            logger.error(f"Failed to read CSV from S3: {e}")
            return None

    def upload_file(self, local_path: str, s3_path: str) -> bool:
        try:
            extra_args = {}
            if local_path.endswith(".json"):
                extra_args["ContentType"] = "application/json"
            elif local_path.endswith(".pt"):
                extra_args["ContentType"] = "application/octet-stream"
            elif local_path.endswith(".npy"):
                extra_args["ContentType"] = "application/octet-stream"

            self.s3.upload_file(local_path, self.bucket, s3_path, ExtraArgs=extra_args)
            logger.info(f"Successfully uploaded {local_path} to s3://{self.bucket}/{s3_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False

class RedisManager:
    def __init__(self):
        self.redis = redis.Redis(
            host=APPConfig.REDIS_HOST,
            port=APPConfig.REDIS_PORT,
            password=APPConfig.REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=5,
            retry_on_timeout=True
        )
        logger.info(f"Connected to Redis at {APPConfig.REDIS_HOST}:{APPConfig.REDIS_PORT}")

    def get_training_job(self) -> Optional[Dict]:
        try:
            result = self.redis.brpop(APPConfig.TRAINING_QUEUE, timeout=1)
            if result:
                _, job_data = result
                return json.loads(job_data)
        except Exception as e:
            logger.error(f"Error getting training job: {e}")
        return None

    def update_status(self, version_id: str, status: Dict):
        try:
            key = f"{APPConfig.MODEL_PREFIX}{version_id}"
            status['timestamp'] = datetime.now().isoformat()
            self.redis.set(key, json.dumps(status), ex=86400)
        except Exception as e:
            logger.error(f"Error updating Redis status: {e}")

class ModelTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model = SentenceTransformer(APPConfig.MODEL_NAME)
        self.model.to(self.device)
        self.redis_manager = RedisManager()
        self.s3_manager = S3Manager()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def update_api_status(self, version_id: str, status: str, error: str = None):
        """Update status via API endpoint synchronously"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
            if error:
                update_data["error"] = error

            url = f"{APPConfig.API_HOST}/model/version/{version_id}/status"
            response = requests.put(url, json=update_data, timeout=5)
            
            if response.status_code != 200:
                logger.error(f"Failed to update status via API: {response.text}")
        except Exception as e:
            logger.error(f"Error updating status via API: {e}")

    def _process_training_data(self, df: pd.DataFrame, config: Dict) -> List[str]:
        try:
            schema = config["schema_mapping"]
            text_parts = []

            for column in ["name_column", "description_column", "category_column"]:
                if schema.get(column) and schema[column] in df.columns:
                    text_parts.append(df[schema[column]].fillna("").astype(str))

            for col in schema.get("custom_columns", []):
                if col["role"] == "training" and col["user_column"] in df.columns:
                    text_parts.append(df[col["user_column"]].fillna("").astype(str))

            if not text_parts:
                raise ValueError("No valid columns found for text processing")

            texts = [" ".join(filter(None, row)) for row in zip(*text_parts)]
            logger.info(f"Processed {len(texts)} text samples")
            return texts

        except Exception as e:
            logger.error(f"Error processing training data: {e}")
            raise

    def _generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        try:
            batch_size = int(batch_size)
            logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")

            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

            embeddings_numpy = embeddings.cpu().numpy()
            logger.info(f"Generated embeddings with shape: {embeddings_numpy.shape}")
            return embeddings_numpy

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def _load_data(self, config: Dict) -> pd.DataFrame:
        try:
            data_source = config["data_source"]
            file_location = data_source["location"]
            
            if file_location.startswith('s3://'):
                file_location = '/'.join(file_location.split('/')[3:])
            
            logger.info(f"Loading CSV from S3 path: {file_location}")
            current_df = self.s3_manager.get_csv_content(file_location)
            
            if current_df is None:
                raise ValueError(f"Failed to load CSV from {file_location}")
                
            logger.info(f"Successfully loaded {len(current_df)} records")
            return current_df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _save_artifacts(self, df: pd.DataFrame, texts: List[str], embeddings: np.ndarray,
                       config: Dict, version_id: str) -> bool:
        try:
            temp_dir = f"/tmp/training/{version_id}"
            os.makedirs(temp_dir, exist_ok=True)

            # Save model state
            model_path = f"{temp_dir}/model.pt"
            torch.save(self.model.state_dict(), model_path)

            # Save embeddings
            embeddings_path = f"{temp_dir}/embeddings.npy"
            np.save(embeddings_path, embeddings)

            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "num_samples": len(texts),
                "embedding_shape": embeddings.shape,
                "model_name": APPConfig.MODEL_NAME,
                "columns": df.columns.tolist(),
                "version": version_id,
                "data_file": config["data_source"]["location"]
            }

            metadata_path = f"{temp_dir}/metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Upload artifacts to S3
            s3_base_path = f"models/{version_id}"
            files_to_upload = {
                "model.pt": model_path,
                "embeddings.npy": embeddings_path,
                "metadata.json": metadata_path,
            }

            for file_name, local_path in files_to_upload.items():
                s3_path = f"{s3_base_path}/{file_name}"
                if not self.s3_manager.upload_file(local_path, s3_path):
                    raise Exception(f"Failed to upload {file_name}")

            logger.info(f"All artifacts saved to S3: {s3_base_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving artifacts: {e}")
            raise

    def train(self, job: Dict) -> bool:
        version_id = job["version_id"]
        config = job["config"]

        try:
            # Update status to processing
            self.update_api_status(version_id, ModelStatus.PROCESSING.value)
            self.redis_manager.update_status(version_id, {"status": ModelStatus.PROCESSING.value})

            # Load and process data
            df = self._load_data(config)
            logger.info(f"Loaded {len(df)} records")

            # Process training data
            texts = self._process_training_data(df, config)
            logger.info(f"Processed {len(texts)} text samples")

            # Get batch size
            batch_size = config.get("training_config", {}).get("batch_size", APPConfig.BATCH_SIZE)

            # Generate embeddings
            embeddings = self._generate_embeddings(texts, batch_size)
            logger.info(f"Generated embeddings of shape {embeddings.shape}")

            # Save artifacts
            success = self._save_artifacts(df, texts, embeddings, config, version_id)

            if success:
                logger.info("Training completed successfully")
                self.update_api_status(version_id, ModelStatus.COMPLETED.value)
                self.redis_manager.update_status(version_id, {"status": ModelStatus.COMPLETED.value})
                return True
            else:
                raise Exception("Failed to save artifacts")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Training failed: {error_msg}")
            self.update_api_status(version_id, ModelStatus.FAILED.value, error_msg)
            self.redis_manager.update_status(
                version_id,
                {"status": ModelStatus.FAILED.value, "error": error_msg}
            )
            return False

class TrainingWorker:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.redis_manager = RedisManager()
        self.should_stop = False
        self.current_job = None
        self._worker_thread = None

    def process_job(self, job: Dict):
        try:
            logger.info(f"Processing job: {job['version_id']}")
            self.trainer.train(job)
        except Exception as e:
            logger.error(f"Error processing job: {e}")

    def _worker_loop(self):
        logger.info("Starting training worker")
        while not self.should_stop:
            try:
                job = self.redis_manager.get_training_job()
                if job:
                    self.current_job = job
                    self.process_job(job)
                    self.current_job = None
                else:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)

    def start(self):
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.should_stop = False
            self._worker_thread = threading.Thread(target=self._worker_loop)
            self._worker_thread.daemon = True
            self._worker_thread.start()

    def stop(self):
        logger.info("Stopping training worker")
        self.should_stop = True
        if self._worker_thread:
            self._worker_thread.join(timeout=5)

# Initialize worker
worker = TrainingWorker()

@app.route("/health")
def health():
    try:
        redis_ok = worker.redis_manager.redis.ping()
        return jsonify({
            "status": "ok",
            "redis": redis_ok
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })
        
        
        
@app.route("/status/<version_id>")
def get_status(version_id):
    """Get training status for a specific version"""
    try:
        status = worker.redis_manager.redis.get(f"{APPConfig.MODEL_PREFIX}{version_id}")
        return jsonify(json.loads(status) if status else {"status": "not_found"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stop", methods=["POST"])
def stop_worker():
    """Stop the training worker"""
    try:
        worker.stop()
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/start", methods=["POST"])
def start_worker():
    """Start the training worker"""
    try:
        worker.start()
        return jsonify({"status": "started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/queue")
def get_queue_status():
    """Get current queue status"""
    try:
        queue_length = worker.redis_manager.redis.llen(APPConfig.TRAINING_QUEUE)
        return jsonify({
            "queue_length": queue_length,
            "current_job": worker.current_job["version_id"] if worker.current_job else None
        })
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
    port = APPConfig.SERVICE_PORT
    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False  # Disable reloader to prevent multiple worker instances
    )
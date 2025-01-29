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
from transformers import AutoTokenizer, AutoModelForCausalLM
 

class ModelStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class AppConfig:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    TRAINING_QUEUE = "training_queue"  # Queue name for training jobs
    MODEL_STATUS_PREFIX = "model_status:"
    SERVICE_PORT = int(os.getenv("SERVICE_PORT", 5001))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 128))
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    S3_BUCKET = os.getenv("S3_BUCKET")
    S3_REGION = os.getenv("AWS_REGION", "us-east-1")
    API_HOST = os.getenv("API_HOST", "http://api:8080")
    AWS_SSL_VERIFY = os.getenv("AWS_SSL_VERIFY", "true")
    AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL")
    DEFAULT_MODEL = "deepseek-coder"
    AVAILABLE_LLM_MODELS = {
        "deepseek-coder": {
            "name": "deepseek-ai/deepseek-coder-1.3b-base",
            "description": "Code-optimized model for technical content and embeddings"
        }
    }
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")
    
    @classmethod
    def get_cache_path(cls, model_path: str) -> str:
        """Get cached model path"""
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
            verify=AppConfig.AWS_SSL_VERIFY.lower() == "true",
            config=config,
        )
        self.bucket = AppConfig.S3_BUCKET
        logger.info(f"Initialized S3Manager with bucket: {self.bucket}")

    def get_csv_content(self, s3_path: str) -> Optional[pd.DataFrame]:
        try:
            if s3_path.startswith("s3://"):
                s3_path = s3_path.split("/", 3)[3]  # Remove 's3://bucket/'

            response = self.s3.get_object(Bucket=self.bucket, Key=s3_path)
            return pd.read_csv(io.BytesIO(response["Body"].read()))
        except Exception as e:
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
            elif local_path.endswith(".csv"):
                extra_args["ContentType"] = "text/csv"

            self.s3.upload_file(local_path, self.bucket, s3_path, ExtraArgs=extra_args)
            logger.info(
                f"Successfully uploaded {local_path} to s3://{self.bucket}/{s3_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False


class RedisManager:
    def __init__(self):
        self.redis = redis.Redis(
            host=AppConfig.REDIS_HOST,
            port=AppConfig.REDIS_PORT,
            password=AppConfig.REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        logger.info(
            f"Connected to Redis at {AppConfig.REDIS_HOST}:{AppConfig.REDIS_PORT}"
        )

    def get_training_job(self) -> Optional[Dict]:
        try:
            result = self.redis.brpop(AppConfig.TRAINING_QUEUE, timeout=1)
            if result:
                _, job_data = result
                return json.loads(job_data)
        except Exception as e:
            logger.error(f"Error getting training job: {e}")
        return None

    def update_status(self, config_id: str, status: Dict):
        try:
            key = f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}"
            status["timestamp"] = datetime.now().isoformat()
            self.redis.set(key, json.dumps(status), ex=86400)  # 24 hour expiry
        except Exception as e:
            logger.error(f"Error updating Redis status: {e}")


class LLMTrainer:
    def __init__(self):
        self.model_name = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_args = {
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_steps": 1000
        }
        self.model = None
        self.tokenizer = None

    def initialize_model(self, model_key):
        """Initialize the model based on the selected key"""
        try:
            if model_key not in AppConfig.AVAILABLE_LLM_MODELS:
                raise ValueError(f"Invalid model key: {model_key}")
            
            self.model_name = AppConfig.AVAILABLE_LLM_MODELS[model_key]["name"]
            logger.info(f"Loading LLM model: {self.model_name}")

            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "output_attentions": False,
                "output_hidden_states": True,  # Always output hidden states
                "return_dict": False  # Return tuple instead of CausalLMOutputWithPast
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            logger.info(f"Successfully loaded LLM model and tokenizer")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing LLM model: {e}")
            raise

    def fine_tune(self, training_data: List[Dict], output_dir: str):
        """Fine-tune the LLM model"""
        try:
            if self.model is None or self.tokenizer is None:
                raise ValueError("Model or tokenizer not initialized")

            logger.info(f"Starting LLM fine-tuning with {len(training_data)} samples")
            
            # Create dataset
            dataset = self._prepare_dataset(training_data)
            
            from transformers import (
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling
            )

            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.training_args["num_train_epochs"],
                per_device_train_batch_size=self.training_args["batch_size"],
                gradient_accumulation_steps=self.training_args["gradient_accumulation_steps"],
                learning_rate=self.training_args["learning_rate"],
                max_steps=self.training_args["max_steps"],
                logging_steps=10,
                save_steps=200,
                save_total_limit=2,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to="none"
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )

            # Train
            logger.info("Starting trainer.train()")
            trainer.train()
            logger.info("Completed trainer.train()")

            # Save model
            logger.info(f"Saving fine-tuned model to {output_dir}")
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save model info
            model_info = {
                "model_type": "llm",
                "base_model": self.model_name,
                "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "fine_tuned": True
            }
            
            with open(f"{output_dir}/model_info.json", "w") as f:
                json.dump(model_info, f)

            return True
            
        except Exception as e:
            logger.error(f"LLM fine-tuning error: {str(e)}")
            raise

    def _prepare_dataset(self, training_data: List[Dict]):
        """Prepare dataset for training"""
        try:
            prompt_template = """### Query: {query}
### Context: {context}
### Response: {response}
"""
            texts = [
                prompt_template.format(**item)
                for item in training_data
            ]
            
            encodings = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )

            # Create proper dataset
            import torch
            dataset = torch.utils.data.TensorDataset(
                encodings["input_ids"],
                encodings["attention_mask"]
            )
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise


class ModelTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.redis_manager = RedisManager()
        self.s3_manager = S3Manager()
        self.llm_trainer = LLMTrainer()

    def update_api_status(
        self, config_id: str, status: str, progress: float = None, error: str = None, model_info: dict = None
    ):
        """Update status via API endpoint with additional model information"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
            if progress is not None:
                update_data["progress"] = progress
            if error:
                update_data["error"] = error
            if model_info:
                update_data.update(model_info)

            url = f"{AppConfig.API_HOST}/config/status/{config_id}"
            response = requests.put(url, json=update_data, timeout=5)

            if response.status_code != 200:
                logger.error(f"Failed to update status via API: {response.text}")

        except Exception as e:
            logger.error(f"Error updating status via API: {e}")

    def _process_training_data(self, df: pd.DataFrame, config: Dict) -> List[str]:
        try:
            schema = config["schema_mapping"]
            text_parts = []

            # Process main columns
            for column in ["name_column", "description_column", "category_column"]:
                if schema.get(column) and schema[column] in df.columns:
                    text_parts.append(df[schema[column]].fillna("").astype(str))

            # Process custom columns marked for training
            for col in schema.get("custom_columns", []):
                if col["role"] == "training" and col["user_column"] in df.columns:
                    text_parts.append(df[col["user_column"]].fillna("").astype(str))

            if not text_parts:
                raise ValueError("No valid columns found for text processing")

            # Combine all text parts with space separator
            texts = [" ".join(filter(None, row)) for row in zip(*text_parts)]
            logger.info(f"Processed {len(texts)} text samples")
            return texts

        except Exception as e:
            logger.error(f"Error processing training data: {e}")
            raise

    def _generate_embeddings(self, texts: List[str], model: LLMTrainer) -> np.ndarray:
        """Generate embeddings using DeepSeek model"""
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = []
            
            # Process in batches
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = model.tokenizer(batch, padding=True, truncation=True, 
                                      return_tensors="pt", max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    # Get hidden states from base model instead of CausalLM output
                    outputs = model.model(
                        **inputs,
                        output_hidden_states=True  # Request hidden states
                    )
                    # Use the last layer's hidden states
                    hidden_states = outputs.hidden_states[-1]  # Get last layer
                    # Mean pool the token embeddings
                    batch_embeddings = hidden_states.mean(dim=1)
                    embeddings.append(batch_embeddings.cpu())
                    
            embeddings = torch.cat(embeddings, dim=0)
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings.numpy()

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def _load_data(self, config: Dict) -> pd.DataFrame:
        try:
            data_source = config["data_source"]
            current_file = data_source["location"]
            data_source = config["data_source"]
            current_file = data_source["location"]

            # Load current data
            current_df = self.s3_manager.get_csv_content(current_file)
            if current_df is None:
                raise ValueError(f"Failed to load current CSV from {current_file}")

            # Handle append mode
            if config["mode"] == "append" and config.get("previous_version"):
                try:
                    # Load previous model data
                    prev_path = f"models/{config['previous_version']}/products.csv"
                    prev_df = self.s3_manager.get_csv_content(prev_path)
                    if prev_df is not None:
                        # Combine datasets
                        current_df = pd.concat([prev_df, current_df], ignore_index=True)
                        logger.info(
                            f"Appended previous data. Total records: {len(current_df)}"
                        )
                except Exception as e:
                    logger.error(f"Error loading previous version data: {e}")
                    raise ValueError(
                        "Failed to load previous version data for append mode"
                    )

            return current_df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _save_artifacts(
        self, df: pd.DataFrame, embeddings: np.ndarray, config: Dict
    ) -> bool:
        """Save model artifacts and metadata"""
        try:
            temp_dir = f"/tmp/training/{config['model_path']}"
            
            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "num_samples": len(df),
                "embedding_shape": embeddings.shape,
                "model_name": AppConfig.AVAILABLE_LLM_MODELS["deepseek-coder"]["name"],
            }
            
            with open(f"{temp_dir}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
                
            # Save products
            df.to_csv(f"{temp_dir}/products.csv", index=False)
            
            # Upload everything to S3
            for file_name in ["metadata.json", "products.csv", "embeddings.npy"]:
                s3_path = f"{config['model_path']}/{file_name}"
                if not self.s3_manager.upload_file(f"{temp_dir}/{file_name}", s3_path):
                    raise Exception(f"Failed to upload {file_name}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {e}")
            raise

    def train(self, job: Dict) -> bool:
        config = job["config"]
        config_id = job["config_id"]

        try:
            # Initialize LLM
            selected_model = "deepseek-coder"  # Using only DeepSeek
            self.llm_trainer.initialize_model(selected_model)
            
            # Load and process data
            df = self._load_data(config)
            texts = self._process_training_data(df, config)
            
            # Generate embeddings using the LLM model
            embeddings = self._generate_embeddings(texts, self.llm_trainer)
            logger.info(f"Generated embeddings with shape {embeddings.shape}")
            
            # Fine-tune the model on product data
            training_data = self.llm_trainer.prepare_training_data(df, config)
            
            # Save everything in one place
            temp_dir = f"/tmp/training/{config['model_path']}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Fine-tune and save model
            self.llm_trainer.fine_tune(training_data, temp_dir)
            
            # Save embeddings alongside the model
            np.save(f"{temp_dir}/embeddings.npy", embeddings)
            
            # Save metadata and products
            success = self._save_artifacts(df, embeddings, config)
            
            if success:
                model_info = {
                    "status": ModelStatus.COMPLETED.value,
                    "progress": 100,
                    "model": selected_model,
                    "model_name": AppConfig.AVAILABLE_LLM_MODELS[selected_model]["name"],
                    "completed_at": datetime.now().isoformat()
                }
                self.update_api_status(config_id, ModelStatus.COMPLETED.value, 
                                     progress=100, model_info=model_info)
                return True

            return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Training failed: {error_msg}")
            self.update_api_status(config_id, ModelStatus.FAILED.value, error=error_msg)
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
            logger.info(f"Processing job for config: {job['config_id']}")
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
            logger.info("Training worker started")

    def stop(self):
        logger.info("Stopping training worker")
        self.should_stop = True
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            logger.info("Training worker stopped")


# Initialize worker
worker = TrainingWorker()


@app.route("/health")
def health():
    """Health check endpoint"""
    try:
        redis_ok = worker.redis_manager.redis.ping()
        return jsonify(
            {
                "status": "healthy",
                "redis": redis_ok,
                "worker": {
                    "running": worker._worker_thread is not None
                    and worker._worker_thread.is_alive(),
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
        # Check Redis first for current status
        status_key = f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}"
        status = worker.redis_manager.redis.get(status_key)

        if status:
            return jsonify(json.loads(status))
        else:
            # If no status in Redis, check current job
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
        queue_length = worker.redis_manager.redis.llen(AppConfig.TRAINING_QUEUE)

        # Get queued jobs
        jobs = []
        if queue_length > 0:
            job_data = worker.redis_manager.redis.lrange(
                AppConfig.TRAINING_QUEUE, 0, -1
            )
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


@app.route("/models")
def get_available_models():
    """Get available LLM models"""
    try:
        return jsonify({
            "models": AppConfig.AVAILABLE_LLM_MODELS
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
    port = AppConfig.SERVICE_PORT
    app.run(
        host="0.0.0.0",
        port=port,
        use_reloader=False,  # Disable reloader to prevent multiple worker instances
    )


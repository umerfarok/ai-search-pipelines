import os
import logging
from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
from pathlib import Path
import redis
from datetime import datetime
import time
import threading
from typing import Dict, List, Optional, Tuple, Union
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import requests
import io
from enum import Enum
import onnx
import onnxruntime as ort
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from dataclasses import dataclass, field


class ModelStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@dataclass
class AppConfig:
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    TRAINING_QUEUE: str = "training_queue"
    MODEL_STATUS_PREFIX: str = "model_status:"
    SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", 5001))
    AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY")
    S3_BUCKET: str = os.getenv("S3_BUCKET")
    S3_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    API_HOST: str = os.getenv("API_HOST", "http://api:8080")
    AWS_SSL_VERIFY: str = os.getenv("AWS_SSL_VERIFY", "true")
    AWS_ENDPOINT_URL: str = os.getenv("AWS_ENDPOINT_URL")
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1

    # Model configurations
    AVAILABLE_LLM_MODELS: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "distilgpt2-product": {
                "name": "distilgpt2",
                "description": "Lightweight GPT model for product search",
            },
            "all-minilm": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",  # This is the correct model ID
                "description": "Efficient embedding model for semantic search",
            },
        }
    )
    DEFAULT_MODEL: str = "distilgpt2-product"
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")
    SHARED_MODELS_DIR: str = os.getenv("SHARED_MODELS_DIR", "/app/shared_models")
    TRANSFORMER_CACHE: str = os.getenv(
        "TRANSFORMER_CACHE", "/app/model_cache/transformers"
    )
    HF_HOME: str = os.getenv("HF_HOME", "/app/model_cache/huggingface")

    @classmethod
    def setup_cache_dirs(cls):
        """Setup cache directories for models"""
        os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.TRANSFORMER_CACHE, exist_ok=True)
        os.makedirs(cls.HF_HOME, exist_ok=True)

        os.environ.update(
            {
                "TORCH_HOME": cls.MODEL_CACHE_DIR,
                "TRANSFORMERS_CACHE": cls.TRANSFORMER_CACHE,
                "HF_HOME": cls.HF_HOME,
                "HF_DATASETS_CACHE": os.path.join(cls.HF_HOME, "datasets"),
            }
        )


class ProductTrainer:
    def __init__(self):
        AppConfig.setup_cache_dirs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.embedding_model = None

    def initialize_models(self, config: Dict):
        """Initialize models based on config"""
        try:
            # Get model settings from config
            print(config)
            model_type = config.get("training_config", {}).get(
                "model_type", "transformer"
            )
            llm_model = config.get("training_config", {}).get("llm_model", "distilgpt2")
            embedding_model = config.get("training_config", {}).get(
                "embedding_model", "all-MiniLM-L6-v2"
            )

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Initialize LLM model with GPU optimization
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": (
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                "low_cpu_mem_usage": True,
            }

            self.model = AutoModelForCausalLM.from_pretrained(llm_model, **model_kwargs)

            # Initialize embedding model with correct identifier
            self.embedding_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.embedding_model.to(self.device)
            logger.info(f"Initialized embedding model on device: {self.device}")

            logger.info(
                f"Successfully initialized models: LLM={llm_model}, Embedding={embedding_model}"
            )
            return True

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def prepare_training_data(self, df: pd.DataFrame, config: Dict) -> Dataset:
        try:
            schema = config["schema_mapping"]
            training_samples = []

            # Use all columns specified in the config
            for _, row in df.iterrows():
                product_info = {
                    "name": str(row[schema["name_column"]]),
                    "description": str(row[schema["description_column"]]),
                    "category": str(row[schema["category_column"]]),
                }

                # Add custom columns from config
                for col in schema.get("custom_columns", []):
                    if col["user_column"] in row:
                        product_info[col["standard_column"]] = str(
                            row[col["user_column"]]
                        )

                queries = self._generate_training_queries(product_info)
                for query, response in queries:
                    training_samples.append(
                        {
                            "query": query,
                            "context": self._format_product_context(product_info),
                            "response": response,
                        }
                    )

            return Dataset.from_dict(
                {
                    "query": [s["query"] for s in training_samples],
                    "context": [s["context"] for s in training_samples],
                    "response": [s["response"] for s in training_samples],
                }
            )

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise

    def fine_tune(self, dataset: Dataset, config: Dict, output_dir: str):
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Models not initialized")

            # Get training parameters from config
            training_config = config.get("training_config", {})
            batch_size = training_config.get("batch_size", 4)
            max_tokens = training_config.get("max_tokens", 512)
            validation_split = training_config.get("validation_split", 0.2)

            # Create output directories
            model_output_dir = os.path.join(output_dir, "llm")
            shared_output_dir = os.path.join(AppConfig.SHARED_MODELS_DIR, output_dir)
            shared_llm_dir = os.path.join(shared_output_dir, "llm")

            os.makedirs(model_output_dir, exist_ok=True)
            os.makedirs(shared_llm_dir, exist_ok=True)

            # Training arguments using config values
            training_args = TrainingArguments(
                output_dir=model_output_dir,
                num_train_epochs=5,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                learning_rate=2e-5,
                optim="adamw_torch_fused",
                max_steps=2000,
                logging_steps=100,
                save_steps=500,
                save_total_limit=2,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to="none",
                save_strategy="steps",
                save_safetensors=False,
                fp16=True
            )

            def tokenize_function(examples):
                prompts = [
                    f"Query: {query}\nContext: {context}\nResponse: {response}"
                    for query, context, response in zip(
                        examples["query"], examples["context"], examples["response"]
                    )
                ]

                return self.tokenizer(
                    prompts,
                    truncation=True,
                    padding="max_length",
                    max_length=max_tokens,
                    return_tensors="pt",
                )

            # Tokenize dataset
            tokenized_dataset = dataset.map(
                tokenize_function, batched=True, remove_columns=dataset.column_names
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    self.tokenizer, mlm=False
                ),
            )

            # Train model
            trainer.train()

            # Save model and tokenizer
            trainer.save_model(model_output_dir)
            self.tokenizer.save_pretrained(model_output_dir)

            # Export to ONNX format
            try:
                self._export_to_onnx(model_output_dir)
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")

            # Copy to shared volume
            self._copy_model_files(model_output_dir, shared_llm_dir)

            return True

        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

    def _export_to_onnx(self, model_dir: str):
        """Export model to ONNX format"""
        try:
            dummy_input = self.tokenizer("Sample text", return_tensors="pt")
            input_names = ["input_ids", "attention_mask"]
            output_names = ["logits"]

            # Export to ONNX
            torch.onnx.export(
                self.model,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                f"{model_dir}/model.onnx",
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                opset_version=12,
            )

            # Verify ONNX model
            onnx_model = onnx.load(f"{model_dir}/model.onnx")
            onnx.checker.check_model(onnx_model)

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise

    def _copy_model_files(self, src_dir: str, dest_dir: str):
        """Copy model files to shared directory"""
        try:
            required_files = [
                "config.json",
                "pytorch_model.bin",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "model.onnx",
            ]

            for file in required_files:
                src_path = os.path.join(src_dir, file)
                dest_path = os.path.join(dest_dir, file)
                if os.path.exists(src_path):
                    os.system(f"cp {src_path} {dest_path}")

        except Exception as e:
            logger.error(f"Error copying model files: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            embeddings = self.embedding_model.encode(
                texts, batch_size=128, show_progress_bar=True, convert_to_tensor=True, device=self.device
            )
            return embeddings.cpu().numpy()

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def _generate_training_queries(self, product_info: Dict) -> List[Tuple[str, str]]:
        """Generate diverse training queries and responses"""
        queries = []

        # Direct product queries
        queries.append(
            (
                f"Do you have {product_info['name']}?",
                f"Yes, we have {product_info['name']}. {product_info.get('description', '')}",
            )
        )

        # Category queries
        queries.append(
            (
                f"What products do you have in {product_info['category']}?",
                f"In {product_info['category']}, we have {product_info['name']}. {product_info.get('description', '')}",
            )
        )

        # Add price queries if available
        if "price" in product_info:
            queries.append(
                (
                    f"How much is {product_info['name']}?",
                    f"{product_info['name']} costs {product_info['price']}.",
                )
            )

        return queries

    def _format_product_context(self, product_info: Dict) -> str:
        """Format product information into context"""
        context = []

        for key, value in product_info.items():
            if value:
                # Convert key from snake_case to Title Case
                formatted_key = " ".join(word.capitalize() for word in key.split("_"))
                context.append(f"{formatted_key}: {value}")

        return "\n".join(context)


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
        """Get next training job from queue"""
        try:
            result = self.redis.brpop(AppConfig.TRAINING_QUEUE, timeout=1)
            if result:
                _, job_data = result
                return json.loads(job_data)
        except Exception as e:
            logger.error(f"Error getting training job: {e}")
        return None

    def update_status(self, config_id: str, status: Dict):
        """Update training status in Redis"""
        try:
            key = f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}"
            status["timestamp"] = datetime.now().isoformat()
            self.redis.set(key, json.dumps(status), ex=86400)  # 24 hour expiry
        except Exception as e:
            logger.error(f"Error updating Redis status: {e}")


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
        """Read CSV content from S3 with retries"""
        for attempt in range(AppConfig.MAX_RETRIES):
            try:
                response = self.s3.get_object(Bucket=self.bucket, Key=s3_path)
                return pd.read_csv(io.BytesIO(response["Body"].read()))
            except Exception as e:
                if attempt == AppConfig.MAX_RETRIES - 1:
                    logger.error(f"Failed to read CSV from S3: {e}")
                    return None
                time.sleep(AppConfig.RETRY_DELAY * (attempt + 1))
        return None

    def upload_file(self, local_path: str, s3_path: str) -> bool:
        """Upload file to S3 with content type and retries"""
        content_types = {
            ".json": "application/json",
            ".pt": "application/octet-stream",
            ".bin": "application/octet-stream",
            ".npy": "application/octet-stream",
            ".csv": "text/csv",
            ".onnx": "application/octet-stream",
        }

        extra_args = {
            "ContentType": content_types.get(
                Path(local_path).suffix, "application/octet-stream"
            )
        }

        for attempt in range(AppConfig.MAX_RETRIES):
            try:
                self.s3.upload_file(
                    local_path, self.bucket, s3_path, ExtraArgs=extra_args
                )
                logger.info(
                    f"Successfully uploaded {local_path} to s3://{self.bucket}/{s3_path}"
                )
                return True
            except Exception as e:
                if attempt == AppConfig.MAX_RETRIES - 1:
                    logger.error(f"Failed to upload to S3: {e}")
                    return False
                time.sleep(AppConfig.RETRY_DELAY * (attempt + 1))
        return False


class ModelTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.redis_manager = RedisManager()
        self.s3_manager = S3Manager()
        self.product_trainer = ProductTrainer()

    def update_api_status(
        self,
        config_id: str,
        status: str,
        progress: float = None,
        error: str = None,
        model_info: dict = None,
    ):
        """Update status via API endpoint"""
        try:
            update_data = {"status": status, "updated_at": datetime.now().isoformat()}

            if progress is not None:
                update_data["progress"] = progress
            if error:
                update_data["error"] = error
            if model_info:
                update_data.update(model_info)

            url = f"{AppConfig.API_HOST}/config/status/{config_id}"

            for attempt in range(AppConfig.MAX_RETRIES):
                try:
                    response = requests.put(url, json=update_data, timeout=5)
                    if response.status_code == 200:
                        return
                    logger.error(f"Failed to update status: {response.text}")
                    if attempt < AppConfig.MAX_RETRIES - 1:
                        time.sleep(AppConfig.RETRY_DELAY * (attempt + 1))
                except Exception as e:
                    if attempt == AppConfig.MAX_RETRIES - 1:
                        logger.error(f"Error updating status via API: {e}")
                    else:
                        time.sleep(AppConfig.RETRY_DELAY * (attempt + 1))

        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def train(self, job: Dict) -> bool:
        config = job["config"]
        config_id = job["config_id"]

        try:
            # Initialize models with config
            self.product_trainer.initialize_models(config)

            # Load data
            df = self._load_data(config)
            if df is None or len(df) == 0:
                raise ValueError("No valid training data found")

            # Update status to processing
            self.update_api_status(
                config_id,
                ModelStatus.PROCESSING.value,
                progress=10,
                model_info={
                    "model": config.get("training_config", {}).get(
                        "llm_model", "distilgpt2"
                    )
                },
            )

            # Prepare data for embeddings
            texts = self._process_training_data(df, config)

            # Generate embeddings
            embeddings = self.product_trainer.generate_embeddings(texts)
            self.update_api_status(config_id, ModelStatus.PROCESSING.value, progress=30)

            # Prepare training data for LLM
            dataset = self.product_trainer.prepare_training_data(df, config)
            self.update_api_status(config_id, ModelStatus.PROCESSING.value, progress=50)

            # Create model directory
            model_dir = os.path.join(AppConfig.MODEL_CACHE_DIR, config["model_path"])
            os.makedirs(model_dir, exist_ok=True)

            # Fine-tune LLM
            success = self.product_trainer.fine_tune(dataset, config, model_dir)
            if not success:
                raise Exception("Model fine-tuning failed")

            self.update_api_status(config_id, ModelStatus.PROCESSING.value, progress=80)

            # Save model artifacts
            np.save(f"{model_dir}/embeddings.npy", embeddings)
            df.to_csv(f"{model_dir}/products.csv", index=False)

            metadata = {
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "num_samples": len(df),
                "embedding_shape": embeddings.shape,
                "models": {
                    "llm": config.get("training_config", {}).get(
                        "llm_model", "distilgpt2"
                    ),
                    "embeddings": config.get("training_config", {}).get(
                        "embedding_model", "all-MiniLM-L6-v2"
                    ),
                },
            }

            with open(f"{model_dir}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Upload to S3
            self._upload_model_files(model_dir, config["model_path"])

            self.update_api_status(
                config_id,
                ModelStatus.COMPLETED.value,
                progress=100,
                model_info={
                    "status": "completed",
                    "model": config.get("training_config", {}).get(
                        "llm_model", "distilgpt2"
                    ),
                },
            )
            return True

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Training failed: {error_msg}")
            self.update_api_status(config_id, ModelStatus.FAILED.value, error=error_msg)
            return False

    def _load_data(self, config: Dict) -> Optional[pd.DataFrame]:
        """Load and combine training data"""
        try:
            data_source = config["data_source"]
            current_file = data_source["location"]

            # Load current data
            current_df = self.s3_manager.get_csv_content(current_file)
            if current_df is None:
                raise ValueError(f"Failed to load current CSV from {current_file}")

            # Handle append mode
            if config["mode"] == "append" and config.get("previous_version"):
                try:
                    prev_path = f"models/{config['previous_version']}/products.csv"
                    prev_df = self.s3_manager.get_csv_content(prev_path)
                    if prev_df is not None:
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

    def _process_training_data(self, df: pd.DataFrame, config: Dict) -> List[str]:
        """Process training data for embedding generation"""
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

            texts = [" ".join(filter(None, row)) for row in zip(*text_parts)]
            logger.info(f"Processed {len(texts)} text samples")
            return texts

        except Exception as e:
            logger.error(f"Error processing training data: {e}")
            raise

    def _upload_model_files(self, local_dir: str, model_path: str) -> bool:
        """Upload all model files to S3"""
        try:
            files_to_upload = [
                "embeddings.npy",
                "metadata.json",
                "products.csv",
                "llm/config.json",
                "llm/pytorch_model.bin",
                "llm/tokenizer.json",
                "llm/model.onnx",
            ]

            for file in files_to_upload:
                local_path = os.path.join(local_dir, file)
                s3_path = f"{model_path}/{file}"

                if os.path.exists(local_path):
                    if not self.s3_manager.upload_file(local_path, s3_path):
                        raise Exception(f"Failed to upload {file}")
                else:
                    logger.warning(f"File not found for upload: {local_path}")

            return True

        except Exception as e:
            logger.error(f"Error uploading model files: {e}")
            raise


class TrainingWorker:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.redis_manager = RedisManager()
        self.should_stop = False
        self.current_job = None
        self._worker_thread = None

    def process_job(self, job: Dict):
        """Process a single training job"""
        try:
            logger.info(f"Processing job for config: {job['config_id']}")
            self.trainer.train(job)
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            self.redis_manager.update_status(
                job["config_id"],
                {
                    "status": ModelStatus.FAILED.value,
                    "error": str(e),
                },
            )

    def _worker_loop(self):
        """Main worker loop"""
        logger.info("Starting training worker loop")
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
        """Start the training worker"""
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.should_stop = False
            self._worker_thread = threading.Thread(target=self._worker_loop)
            self._worker_thread.daemon = True
            self._worker_thread.start()
            logger.info("Training worker started")

    def stop(self):
        """Stop the training worker"""
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
                "device": worker.trainer.device,
            }
        )
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/status/<config_id>")
def get_status(config_id):
    """Get training status"""
    try:
        status_key = f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}"
        status = worker.redis_manager.redis.get(status_key)

        if status:
            return jsonify(json.loads(status))

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
    """Get available LLM models info"""
    try:
        return jsonify(
            {
                "models": AppConfig.AVAILABLE_LLM_MODELS,
                "default": AppConfig.DEFAULT_MODEL,
                "device": worker.trainer.device,
            }
        )
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

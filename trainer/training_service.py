import os
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from datasets import Dataset
from dataclasses import dataclass, field
import multiprocessing


multiprocessing.set_start_method("spawn", force=True)


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

    # Default PEFT configuration
    PEFT_CONFIG = {
        "r": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    # Model configurations
    AVAILABLE_LLM_MODELS: Dict[str, Dict[str, str]] = field(
        default_factory=lambda: {
            "distilgpt2-product": {
                "name": "distilgpt2",
                "description": "Fast, lightweight model for product search",
            },
            "all-minilm-l6": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
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
        for directory in [
            cls.MODEL_CACHE_DIR,
            cls.TRANSFORMER_CACHE,
            cls.HF_HOME,
            os.path.join(cls.HF_HOME, "datasets"),
        ]:
            os.makedirs(directory, exist_ok=True)

        os.environ.update(
            {
                "TORCH_HOME": cls.MODEL_CACHE_DIR,
                "TRANSFORMERS_CACHE": cls.TRANSFORMER_CACHE,
                "HF_HOME": cls.HF_HOME,
                "HF_DATASETS_CACHE": os.path.join(cls.HF_HOME, "datasets"),
            }
        )


class FastProductTrainer:
    def __init__(self):
        AppConfig.setup_cache_dirs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.embedding_model = None

    def initialize_models(self, config: Dict):
        """Initialize models with PEFT configuration"""
        try:
            model_name = config.get("training_config", {}).get(
                "llm_model", "distilgpt2"
            )

            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Initialize base model
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": (
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                "low_cpu_mem_usage": True,
            }

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )

            # Setup PEFT configuration
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=AppConfig.PEFT_CONFIG["r"],
                lora_alpha=AppConfig.PEFT_CONFIG["lora_alpha"],
                lora_dropout=AppConfig.PEFT_CONFIG["lora_dropout"],
                bias=AppConfig.PEFT_CONFIG["bias"],
            )

            # Create PEFT model
            self.peft_model = get_peft_model(self.model, peft_config)
            self.peft_model.print_trainable_parameters()

            # Initialize embedding model
            self.embedding_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.embedding_model.to(self.device)

            logger.info("Successfully initialized models with PEFT")
            return True

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def prepare_training_data(self, df: pd.DataFrame, config: Dict) -> Dataset:
        """Prepare training data with proper validation"""
        try:
            if len(df) == 0:
                raise ValueError("Empty dataframe provided")

            schema = config["schema_mapping"]
            required_columns = ["name_column", "description_column", "category_column"]

            # Validate schema
            for col in required_columns:
                if not schema.get(col) or schema[col] not in df.columns:
                    raise ValueError(f"Missing required column: {schema.get(col)}")

            training_samples = []

            # Process each product
            for _, row in df.iterrows():
                product_info = {
                    "name": str(row[schema["name_column"]]),
                    "description": str(row[schema["description_column"]]),
                    "category": str(row[schema["category_column"]]),
                }

                # Add custom columns
                for col in schema.get("custom_columns", []):
                    if col["user_column"] in row:
                        product_info[col["standard_column"]] = str(
                            row[col["user_column"]]
                        )

                queries = self._generate_training_queries(product_info)
                context = self._format_product_context(product_info)

                for query, response in queries:
                    training_samples.append(
                        {"query": query, "context": context, "response": response}
                    )

            if len(training_samples) == 0:
                raise ValueError("No valid training samples generated")

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

    def fast_train(self, dataset: Dataset, config: Dict, output_dir: str):
        """Fast training implementation with proper resource management"""
        if not dataset:
            raise ValueError("No dataset provided for training")

        try:
            if not self.peft_model or not self.tokenizer:
                raise ValueError("Models not initialized")

            # Get training parameters
            training_config = config.get("training_config", {})
            batch_size = training_config.get("batch_size", 4)
            max_tokens = training_config.get("max_tokens", 512)

            # Setup directories
            model_output_dir = os.path.join(output_dir, "llm")
            shared_model_dir = os.path.join(AppConfig.SHARED_MODELS_DIR, output_dir)
            shared_llm_dir = os.path.join(shared_model_dir, "llm")

            for dir_path in [model_output_dir, shared_model_dir, shared_llm_dir]:
                os.makedirs(dir_path, exist_ok=True)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=model_output_dir,
                num_train_epochs=1,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=1,
                learning_rate=1e-4,
                max_steps=100,
                logging_steps=10,
                save_steps=50,
                fp16=torch.cuda.is_available(),
                optim="adamw_torch",
                remove_unused_columns=False,
                push_to_hub=False,
                report_to="none",
                dataloader_num_workers=(
                    0 if torch.cuda.is_available() else 4
                ),  # Prevent CUDA fork issues
            )

            # Tokenize dataset
            tokenized_dataset = self._tokenize_dataset(dataset, max_tokens)

            # Initialize trainer
            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    self.tokenizer, mlm=False
                ),
            )

            # Train model
            trainer.train()

            # Save and validate
            self._save_and_validate_model(trainer, model_output_dir, shared_llm_dir)

            return True

        except Exception as e:
            logger.error(f"Training error: {e}")
            self._cleanup_failed_training(model_output_dir, shared_llm_dir)
            raise

    def _tokenize_dataset(self, dataset: Dataset, max_tokens: int) -> Dataset:
        """Tokenize dataset with proper CUDA handling"""
        try:

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

            # Use single process if CUDA is available to avoid fork issues
            if torch.cuda.is_available():
                logger.info("CUDA detected - using single process for tokenization")
                num_proc = None
                torch.multiprocessing.set_start_method("spawn", force=True)
            else:
                logger.info("Using multiple processes for tokenization")
                num_proc = os.cpu_count()

            return dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names,
                num_proc=num_proc,
            )

        except Exception as e:
            logger.error(f"Error tokenizing dataset: {e}")
            raise

    def _save_and_validate_model(self, trainer, model_dir: str, shared_dir: str):
        """Save and validate model with comprehensive checks"""
        try:
            # Save model
            trainer.save_model(model_dir)
            self.tokenizer.save_pretrained(model_dir)

            # Verify files
            required_files = [
                "adapter_config.json",
                "adapter_model.bin",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ]

            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(model_dir, file)):
                    missing_files.append(file)

            if missing_files:
                raise ValueError(f"Missing required model files: {missing_files}")

            # Validate model by attempting to reload
            try:
                test_model = PeftModel.from_pretrained(
                    self.model, model_dir, device_map="auto"
                )
                test_tokenizer = AutoTokenizer.from_pretrained(model_dir)

                logger.info("Successfully validated model reload")

                # Copy to shared directory
                for file in os.listdir(model_dir):
                    shutil.copy2(
                        os.path.join(model_dir, file), os.path.join(shared_dir, file)
                    )

            except Exception as e:
                raise ValueError(f"Model validation failed: {e}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def _cleanup_failed_training(self, *dirs):
        """Clean up resources after failed training"""
        for dir_path in dirs:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                    logger.info(f"Cleaned up directory: {dir_path}")
            except Exception as e:
                logger.error(f"Failed to clean up {dir_path}: {e}")

    def _format_product_context(self, product_info: Dict) -> str:
        """Format product information into context string"""
        try:
            context = []

            # Add main product information
            context.append(f"Product: {product_info['name']}")
            if product_info.get("description"):
                context.append(f"Description: {product_info['description']}")
            context.append(f"Category: {product_info['category']}")

            # Add custom fields
            for key, value in product_info.items():
                if key not in ["name", "description", "category"] and value:
                    formatted_key = " ".join(
                        word.capitalize() for word in key.split("_")
                    )
                    context.append(f"{formatted_key}: {value}")

            return "\n".join(context)

        except Exception as e:
            logger.error(f"Error formatting product context: {e}")
            raise

    def _generate_training_queries(self, product_info: Dict) -> List[Tuple[str, str]]:
        """Generate diverse training queries and responses"""
        queries = []

        # Add basic queries
        queries.extend(
            [
                (
                    f"Do you have {product_info['name']}?",
                    f"Yes, we have {product_info['name']}. {product_info.get('description', '')}",
                ),
                (
                    f"Tell me about {product_info['name']}",
                    f"{product_info['name']} is {product_info.get('description', '')}. It belongs to the {product_info['category']} category.",
                ),
                (
                    f"What products do you have in {product_info['category']}?",
                    f"In {product_info['category']}, we have {product_info['name']}. {product_info.get('description', '')}",
                ),
            ]
        )

        # Add custom field queries
        for key, value in product_info.items():
            if key not in ["name", "description", "category"] and value:
                queries.append(
                    (
                        f"What is the {key} of {product_info['name']}?",
                        f"The {key} of {product_info['name']} is {value}.",
                    )
                )

        return queries

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using parallel processing"""
        try:
            if not texts:
                raise ValueError("No texts provided for embedding generation")

            # Process in batches
            batch_size = 32
            embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                with torch.no_grad():
                    batch_embeddings = self.embedding_model.encode(
                        batch, convert_to_tensor=True, show_progress_bar=False
                    )
                    if isinstance(batch_embeddings, torch.Tensor):
                        batch_embeddings = batch_embeddings.cpu().numpy()
                    embeddings.append(batch_embeddings)

            return np.vstack(embeddings)

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


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
        """Get next training job from queue with retries"""
        for attempt in range(AppConfig.MAX_RETRIES):
            try:
                result = self.redis.brpop(AppConfig.TRAINING_QUEUE, timeout=1)
                if result:
                    _, job_data = result
                    return json.loads(job_data)
                return None
            except Exception as e:
                if attempt == AppConfig.MAX_RETRIES - 1:
                    logger.error(f"Error getting training job: {e}")
                else:
                    time.sleep(AppConfig.RETRY_DELAY * (attempt + 1))
        return None

    def update_status(self, config_id: str, status: Dict):
        """Update training status in Redis with retries"""
        key = f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}"
        status["timestamp"] = datetime.now().isoformat()

        for attempt in range(AppConfig.MAX_RETRIES):
            try:
                self.redis.set(key, json.dumps(status), ex=86400)  # 24 hour expiry
                return
            except Exception as e:
                if attempt == AppConfig.MAX_RETRIES - 1:
                    logger.error(f"Error updating Redis status: {e}")
                else:
                    time.sleep(AppConfig.RETRY_DELAY * (attempt + 1))


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
        """Upload file to S3 with content type detection and retries"""
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


class FastModelTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.redis_manager = RedisManager()
        self.s3_manager = S3Manager()
        self.product_trainer = FastProductTrainer()

    def update_api_status(
        self,
        config_id: str,
        status: str,
        progress: float = None,
        error: str = None,
        model_info: dict = None,
    ):
        """Update status via API with retries"""
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
        """Main training function with proper error handling and status updates"""
        config = job["config"]
        config_id = job["config_id"]

        try:
            # Initialize models
            self.product_trainer.initialize_models(config)

            # Load and validate data
            df = self._load_data(config)
            if df is None or len(df) == 0:
                raise ValueError("No valid training data found")

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

            # Process data and generate embeddings
            texts = self._process_training_data(df, config)
            embeddings = self.product_trainer.generate_embeddings(texts)
            self.update_api_status(config_id, ModelStatus.PROCESSING.value, progress=30)

            # Prepare and train model
            dataset = self.product_trainer.prepare_training_data(df, config)
            self.update_api_status(config_id, ModelStatus.PROCESSING.value, progress=50)

            # Setup directories
            cache_dir = os.path.join(AppConfig.MODEL_CACHE_DIR, config["model_path"])
            shared_dir = os.path.join(AppConfig.SHARED_MODELS_DIR, config["model_path"])
            os.makedirs(os.path.join(cache_dir, "llm"), exist_ok=True)
            os.makedirs(os.path.join(shared_dir, "llm"), exist_ok=True)

            # Train model
            success = self.product_trainer.fast_train(dataset, config, cache_dir)
            if not success:
                raise Exception("Model training failed")

            # Save model files and upload to S3
            self._save_model_files(cache_dir, shared_dir, embeddings, df, config)
            if not self._upload_model_files(shared_dir, config["model_path"]):
                raise Exception("Failed to upload model files")

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

    def _save_model_files(
        self,
        cache_dir: str,
        shared_dir: str,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        config: Dict,
    ):
        """Save model files with enhanced verification"""
        for target_dir in [cache_dir, shared_dir]:
            try:
                logger.info(f"Saving model files to {target_dir}")
                os.makedirs(target_dir, exist_ok=True)
                llm_dir = os.path.join(target_dir, "llm")
                os.makedirs(llm_dir, exist_ok=True)

                # Save embeddings and data
                np.save(os.path.join(target_dir, "embeddings.npy"), embeddings)
                df.to_csv(os.path.join(target_dir, "products.csv"), index=False)

                # Save metadata
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "config": config,
                    "num_samples": len(df),
                    "embedding_shape": embeddings.shape,
                    "models": {
                        "llm": config.get("training_config", {}).get(
                            "llm_model", "distilgpt2"
                        ),
                        "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
                    },
                }

                with open(os.path.join(target_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)

                # Verify files
                required_files = ["embeddings.npy", "products.csv", "metadata.json"]
                for file in required_files:
                    if not os.path.exists(os.path.join(target_dir, file)):
                        raise ValueError(f"Missing required file: {file}")

            except Exception as e:
                logger.error(f"Error saving files to {target_dir}: {e}")
                raise

    def _upload_model_files(self, local_dir: str, model_path: str) -> bool:
        """Upload model files to S3 with verification"""
        try:
            files_to_upload = [
                "embeddings.npy",
                "metadata.json",
                "products.csv",
                "llm/adapter_config.json",
                "llm/adapter_model.bin",
                "llm/tokenizer.json",
            ]

            for file in files_to_upload:
                local_path = os.path.join(local_dir, file)
                s3_path = f"{model_path}/{file}"

                if not os.path.exists(local_path):
                    raise ValueError(f"Required file missing: {file}")

                if not self.s3_manager.upload_file(local_path, s3_path):
                    raise Exception(f"Failed to upload {file}")

            return True

        except Exception as e:
            logger.error(f"Error uploading model files: {e}")
            return False

    def _load_data(self, config: Dict) -> Optional[pd.DataFrame]:
        """Load training data with validation"""
        try:
            data_source = config["data_source"]
            current_file = data_source["location"]

            current_df = self.s3_manager.get_csv_content(current_file)
            if current_df is None:
                raise ValueError(f"Failed to load CSV from {current_file}")

            # Handle append mode
            if config["mode"] == "append" and config.get("previous_version"):
                prev_path = f"models/{config['previous_version']}/products.csv"
                prev_df = self.s3_manager.get_csv_content(prev_path)
                if prev_df is not None:
                    current_df = pd.concat([prev_df, current_df], ignore_index=True)

            return current_df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _process_training_data(self, df: pd.DataFrame, config: Dict) -> List[str]:
        """Process training data for embeddings"""
        try:
            schema = config["schema_mapping"]
            text_parts = []

            # Process main columns
            for column in ["name_column", "description_column", "category_column"]:
                if schema.get(column) and schema[column] in df.columns:
                    text_parts.append(df[schema[column]].fillna("").astype(str))

            # Process custom columns
            for col in schema.get("custom_columns", []):
                if col["role"] == "training" and col["user_column"] in df.columns:
                    text_parts.append(df[col["user_column"]].fillna("").astype(str))

            if not text_parts:
                raise ValueError("No valid columns found for text processing")

            return [" ".join(filter(None, row)) for row in zip(*text_parts)]

        except Exception as e:
            logger.error(f"Error processing training data: {e}")
            raise


class TrainingWorker:
    def __init__(self):
        self.trainer = FastModelTrainer()
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
                job["config_id"], {"status": ModelStatus.FAILED.value, "error": str(e)}
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


# Flask routes
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
                            "config": {
                                "name": parsed_job["config"].get("name", "Unnamed"),
                                "model_path": parsed_job["config"].get(
                                    "model_path", "unknown"
                                ),
                                "mode": parsed_job["config"].get("mode", "replace"),
                            },
                        }
                    )
                except json.JSONDecodeError:
                    continue

        return jsonify(
            {
                "queue_length": queue_length,
                "current_job": (
                    {
                        "config_id": worker.current_job["config_id"],
                        "config": worker.current_job["config"],
                    }
                    if worker.current_job
                    else None
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

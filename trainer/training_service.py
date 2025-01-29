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
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

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

class AppConfig:
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
    TRAINING_QUEUE = "training_queue"
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
    
    # Updated model configurations
    AVAILABLE_LLM_MODELS = {
        "gpt2-product": {
            "name": "gpt2",
            "description": "Fine-tuned GPT-2 model for product search"
        },
        "all-mpnet-base": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "description": "Powerful embedding model for semantic search"
        }
    }
    DEFAULT_MODEL = "gpt2-product"
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")
    SHARED_MODELS_DIR = os.getenv("SHARED_MODELS_DIR", "/app/shared_models")
    TRANSFORMER_CACHE = os.getenv("TRANSFORMER_CACHE", "/app/model_cache/transformers")
    HF_HOME = os.getenv("HF_HOME", "/app/model_cache/huggingface")
    
    @classmethod
    def get_cache_path(cls, model_path: str) -> str:
        safe_path = model_path.replace("/", "_").replace("\\", "_")
        return os.path.join(cls.MODEL_CACHE_DIR, safe_path)
    
    @classmethod
    def get_model_path(cls, model_id: str) -> str:
        return os.path.join(cls.MODEL_CACHE_DIR, model_id)

    @classmethod
    def setup_cache_dirs(cls):
        """Setup cache directories for models"""
        os.makedirs(cls.MODEL_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.TRANSFORMER_CACHE, exist_ok=True)
        os.makedirs(cls.HF_HOME, exist_ok=True)
        
        # Set environment variables for model caching
        os.environ["TORCH_HOME"] = cls.MODEL_CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = cls.TRANSFORMER_CACHE
        os.environ["HF_HOME"] = cls.HF_HOME
        os.environ["HF_DATASETS_CACHE"] = os.path.join(cls.HF_HOME, "datasets")


class ProductTrainer:
    def __init__(self):
        # Set up cache directories before initializing models
        AppConfig.setup_cache_dirs()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_tokenizer = None
        self.llm_model = None
        self.embedding_model = None
        self.training_args = {
            "num_train_epochs": 5,  # Increased epochs for better learning
            "learning_rate": 2e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_steps": 2000,  # Increased steps
            "save_steps": 500,
            "logging_steps": 100
        }

    def initialize_models(self):
        """Initialize both LLM and embedding models"""
        try:
            # Initialize LLM (GPT-2)
            model_name = AppConfig.AVAILABLE_LLM_MODELS["gpt2-product"]["name"]
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(
                AppConfig.AVAILABLE_LLM_MODELS["all-mpnet-base"]["name"]
            )
            self.embedding_model.to(self.device)
            
            logger.info(f"Successfully initialized models")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def prepare_training_data(self, df: pd.DataFrame, config: Dict) -> Dataset:
        """Prepare training data for LLM fine-tuning"""
        try:
            schema = config["schema_mapping"]
            training_samples = []
            
            # Enhanced training samples with more diverse patterns
            for _, row in df.iterrows():
                name = str(row[schema["name_column"]])
                description = str(row[schema["description_column"]])
                category = str(row[schema["category_column"]])
                
                # Create product context with all available information
                product_info = {
                    "name": name,
                    "description": description,
                    "category": category
                }
                
                # Add custom columns
                for col in schema.get("custom_columns", []):
                    if col["user_column"] in row:
                        product_info[col["standard_column"]] = str(row[col["user_column"]])
                
                # Generate diverse training samples
                queries = self._generate_training_queries(product_info)
                for query, response in queries:
                    training_samples.append({
                        "query": query,
                        "context": self._format_product_context(product_info),
                        "response": response
                    })
            
            return Dataset.from_dict({
                "query": [s["query"] for s in training_samples],
                "context": [s["context"] for s in training_samples],
                "response": [s["response"] for s in training_samples]
            })

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise

    def _generate_training_queries(self, product_info: Dict) -> List[Tuple[str, str]]:
        """Generate diverse training queries and responses"""
        queries = []
        
        # Direct product queries
        queries.append((
            f"Do you have {product_info['name']}?",
            f"Yes, we have {product_info['name']}. {product_info['description']}"
        ))
        
        # Use case queries
        use_cases = self._extract_use_cases(product_info['description'])
        for use_case in use_cases:
            queries.append((
                f"I need something to {use_case}",
                f"For {use_case}, I recommend {product_info['name']}. {product_info['description']}"
            ))
        
        # Category-based queries
        queries.append((
            f"What products do you have in {product_info['category']}?",
            f"In {product_info['category']}, we have {product_info['name']}. {product_info['description']}"
        ))
        
        return queries

    def _format_product_context(self, product_info: Dict) -> str:
        """Format product information into a structured context string"""
        context = []
        
        # Add basic product information
        if product_info.get("name"):
            context.append(f"Product Name: {product_info['name']}")
            
        if product_info.get("description"):
            context.append(f"Description: {product_info['description']}")
            
        if product_info.get("category"):
            context.append(f"Category: {product_info['category']}")
            
        # Add any additional metadata
        other_info = {k: v for k, v in product_info.items() 
                     if k not in ["name", "description", "category"]}
                     
        if other_info:
            context.append("Additional Information:")
            for key, value in other_info.items():
                context.append(f"- {key}: {value}")
                
        return "\n".join(context)

    def _extract_use_cases(self, text: str) -> List[str]:
        """Extract potential use cases from product description"""
        use_cases = []
        
        # Common action verbs and tasks
        action_verbs = ["clean", "wash", "cook", "fry", "store", "organize", 
                       "protect", "secure", "maintain", "repair", "fix", "remove",
                       "control", "kill", "eliminate", "prevent", "treat"]
                       
        # Look for patterns like "used for...", "designed to...", "helps..."
        text = text.lower()
        words = text.split()
        
        for i, word in enumerate(words):
            # Check for action verbs
            if word in action_verbs:
                # Try to capture the context around the verb
                start = max(0, i - 3)
                end = min(len(words), i + 4)
                use_case = " ".join(words[start:end])
                use_cases.append(use_case)
                
            # Look for common phrases
            if word in ["for", "to"] and i > 0:
                if words[i-1] in ["used", "designed", "made", "helps", "ideal"]:
                    start = i
                    end = min(len(words), i + 4)
                    use_case = " ".join(words[start:end])
                    use_cases.append(use_case)
        
        # Make unique and clean up
        use_cases = list(set(use_cases))
        return [uc.strip() for uc in use_cases if len(uc.strip()) > 5]

    def fine_tune(self, dataset: Dataset, output_dir: str):
        """Fine-tune the LLM model"""
        try:
            if self.llm_model is None or self.llm_tokenizer is None:
                raise ValueError("Models not initialized")

            # Create model output directory early
            model_output_dir = os.path.join(output_dir, "llm")
            os.makedirs(model_output_dir, exist_ok=True)

            logger.info(f"Starting LLM fine-tuning with {len(dataset)} samples")
            
            training_args = TrainingArguments(
                output_dir=model_output_dir,  # Changed to model_output_dir
                num_train_epochs=self.training_args["num_train_epochs"],
                per_device_train_batch_size=self.training_args["batch_size"],
                gradient_accumulation_steps=self.training_args["gradient_accumulation_steps"],
                learning_rate=self.training_args["learning_rate"],
                max_steps=self.training_args["max_steps"],
                logging_steps=self.training_args["logging_steps"],
                save_steps=self.training_args["save_steps"],
                save_total_limit=2,
                remove_unused_columns=False,
                push_to_hub=False,
                report_to="none",
                save_strategy="steps",  # Add this
                save_safetensors=False,  # Add this to use .bin format
            )

            def tokenize_function(examples):
                # Create prompt format
                prompts = [
                    f"Query: {query}\nContext: {context}\nResponse: {response}"
                    for query, context, response in zip(
                        examples["query"], 
                        examples["context"], 
                        examples["response"]
                    )
                ]
                
                return self.llm_tokenizer(
                    prompts,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )

            # Tokenize dataset
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.llm_model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    self.llm_tokenizer,
                    mlm=False
                ),
            )

            # Train
            trainer.train()
            
            # Explicitly save the model and verify files
            logger.info(f"Saving model to {model_output_dir}")
            trainer.save_model(model_output_dir)
            
            # Save tokenizer
            self.llm_tokenizer.save_pretrained(model_output_dir)
            
            # Verify the model files exist
            required_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
            missing_files = []
            for file in required_files:
                file_path = os.path.join(model_output_dir, file)
                if not os.path.exists(file_path):
                    missing_files.append(file)
                else:
                    logger.info(f"Verified file exists: {file_path}")
            
            if missing_files:
                raise Exception(f"Required model files missing after save: {', '.join(missing_files)}")
            
            # Save model info
            model_info = {
                "model_type": "llm",
                "base_model": AppConfig.AVAILABLE_LLM_MODELS["gpt2-product"]["name"],
                "embedding_model": AppConfig.AVAILABLE_LLM_MODELS["all-mpnet-base"]["name"],
                "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "fine_tuned": True,
                "model_files": [f for f in os.listdir(model_output_dir) if os.path.isfile(os.path.join(model_output_dir, f))]
            }
            
            with open(os.path.join(model_output_dir, "model_info.json"), "w") as f:
                json.dump(model_info, f, indent=2)

            return True
            
        except Exception as e:
            logger.error(f"LLM fine-tuning error: {str(e)}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using the sentence transformer model"""
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=128,
                show_progress_bar=True,
                convert_to_tensor=True
            )
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


class ModelTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.redis_manager = RedisManager()
        self.s3_manager = S3Manager()
        self.product_trainer = ProductTrainer()
        logger.info(f"Using device: {self.device}")
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

            # Combine all text parts
            texts = [" ".join(filter(None, row)) for row in zip(*text_parts)]
            logger.info(f"Processed {len(texts)} text samples")
            return texts

        except Exception as e:
            logger.error(f"Error processing training data: {e}")
            raise

    def train(self, job: Dict) -> bool:
        config = job["config"]
        config_id = job["config_id"]

        try:
            # Initialize models
            self.product_trainer.initialize_models()
            
            # Load and process data
            df = self._load_data(config)
            
            # Prepare data for embeddings
            texts = self._process_training_data(df, config)
            
            # Generate embeddings
            embeddings = self.product_trainer.generate_embeddings(texts)
            
            # Prepare training data for LLM
            dataset = self.product_trainer.prepare_training_data(df, config)
            
            # Create temporary directory for model artifacts
            temp_dir = os.path.join(AppConfig.MODEL_CACHE_DIR, config['model_path'])
            os.makedirs(temp_dir, exist_ok=True)
            
            # Fine-tune LLM with verified output
            success = self.product_trainer.fine_tune(dataset, temp_dir)
            
            if success:
                # Save embeddings
                np.save(f"{temp_dir}/embeddings.npy", embeddings)
                
                # Save products data
                df.to_csv(f"{temp_dir}/products.csv", index=False)
                
                # Save metadata
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "config": config,
                    "num_samples": len(df),
                    "embedding_shape": embeddings.shape,
                    "models": {
                        "llm": AppConfig.AVAILABLE_LLM_MODELS["gpt2-product"]["name"],
                        "embeddings": AppConfig.AVAILABLE_LLM_MODELS["all-mpnet-base"]["name"]
                    }
                }
                
                with open(f"{temp_dir}/metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Upload to S3
                for item in ["metadata.json", "products.csv", "embeddings.npy"]:
                    s3_path = f"{config['model_path']}/{item}"
                    if not self.s3_manager.upload_file(f"{temp_dir}/{item}", s3_path):
                        raise Exception(f"Failed to upload {item}")
                
                # Upload LLM model files with retries
                llm_files = [
                    "config.json",
                    "pytorch_model.bin", 
                    "training_args.bin",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "model_info.json"
                ]

                # Try to upload each file with retries
                max_retries = 3
                for file in llm_files:
                    local_path = f"{temp_dir}/llm/{file}"
                    s3_path = f"{config['model_path']}/llm/{file}"
                    
                    if os.path.exists(local_path):
                        retry_count = 0
                        while retry_count < max_retries:
                            try:
                                logger.info(f"Uploading {local_path} to {s3_path}")
                                if not self.s3_manager.upload_file(local_path, s3_path):
                                    raise Exception(f"Failed to upload {file}")
                                break
                            except Exception as e:
                                retry_count += 1
                                if retry_count == max_retries:
                                    raise Exception(f"Failed to upload {file} after {max_retries} attempts: {str(e)}")
                                time.sleep(1)  # Wait before retrying
                    else:
                        logger.warning(f"File not found: {local_path}")
                        
                # Verify all required files were uploaded
                required_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
                for file in required_files:
                    s3_path = f"{config['model_path']}/llm/{file}"
                    try:
                        self.s3_manager.s3.head_object(Bucket=self.s3_manager.bucket, Key=s3_path)
                    except Exception as e:
                        raise Exception(f"Required file {file} not found in S3 after upload")
                
                self.update_api_status(
                    config_id,
                    ModelStatus.COMPLETED.value,
                    progress=100,
                    model_info={
                        "status": "completed",
                        "model": AppConfig.DEFAULT_MODEL
                    }
                )
                return True
                
            return False

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Training failed: {error_msg}")
            self.update_api_status(
                config_id,
                ModelStatus.FAILED.value,
                error=error_msg
            )
            return False

    def _load_data(self, config: Dict) -> pd.DataFrame:
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
                        logger.info(f"Appended previous data. Total records: {len(current_df)}")
                except Exception as e:
                    logger.error(f"Error loading previous version data: {e}")
                    raise ValueError("Failed to load previous version data for append mode")

            return current_df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
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
        logger.info(f"Connected to Redis at {AppConfig.REDIS_HOST}:{AppConfig.REDIS_PORT}")

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
        """Read CSV content from S3"""
        try:
            if s3_path.startswith("s3://"):
                s3_path = s3_path.split("/", 3)[3]  # Remove 's3://bucket/'

            response = self.s3.get_object(Bucket=self.bucket, Key=s3_path)
            return pd.read_csv(io.BytesIO(response["Body"].read()))
        except Exception as e:
            logger.error(f"Failed to read CSV from S3: {e}")
            return None

    def upload_file(self, local_path: str, s3_path: str) -> bool:
        """Upload file to S3 with appropriate content type"""
        try:
            extra_args = {}
            if local_path.endswith(".json"):
                extra_args["ContentType"] = "application/json"
            elif local_path.endswith(".pt") or local_path.endswith(".bin"):
                extra_args["ContentType"] = "application/octet-stream"
            elif local_path.endswith(".npy"):
                extra_args["ContentType"] = "application/octet-stream"
            elif local_path.endswith(".csv"):
                extra_args["ContentType"] = "text/csv"

            self.s3.upload_file(local_path, self.bucket, s3_path, ExtraArgs=extra_args)
            logger.info(f"Successfully uploaded {local_path} to s3://{self.bucket}/{s3_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False


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
            logger.info(f"Config: {job}")
            self.trainer.train(job)
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            # Update status with error
            self.redis_manager.update_status(
                job["config_id"],
                {
                    "status": ModelStatus.FAILED.value,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )

    def _worker_loop(self):
        """Main worker loop for processing training jobs"""
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
        return jsonify({
            "status": "healthy",
            "redis": redis_ok,
            "worker": {
                "running": worker._worker_thread is not None and worker._worker_thread.is_alive(),
                "current_job": worker.current_job["config_id"] if worker.current_job else None,
            },
        })
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
                return jsonify({
                    "status": ModelStatus.PROCESSING.value,
                    "timestamp": datetime.now().isoformat(),
                })
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
            job_data = worker.redis_manager.redis.lrange(AppConfig.TRAINING_QUEUE, 0, -1)
            for job in job_data:
                try:
                    parsed_job = json.loads(job)
                    jobs.append({
                        "config_id": parsed_job["config_id"],
                        "timestamp": parsed_job.get("timestamp", "unknown"),
                    })
                except json.JSONDecodeError:
                    continue

        return jsonify({
            "queue_length": queue_length,
            "current_job": worker.current_job["config_id"] if worker.current_job else None,
            "queued_jobs": jobs,
        })
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
            "models": AppConfig.AVAILABLE_LLM_MODELS,
            "default": AppConfig.DEFAULT_MODEL
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
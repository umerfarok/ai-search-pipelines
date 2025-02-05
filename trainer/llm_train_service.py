import os
import logging
import time
from typing import Dict, List, Optional, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import redis
from pymongo import MongoClient
from bson import ObjectId
import threading
from flask import Flask, request, jsonify
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from config import AppConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class ModelTracker:
    def __init__(self):
        self.mongo_client = MongoClient(AppConfig.MONGO_URI)
        self.db = self.mongo_client[AppConfig.MONGO_DB]
        self.models_collection = self.db.llm_models
        self.versions_collection = self.db.model_versions

    def create_model_version(self, config: Dict) -> str:
        """Create a new model version entry"""
        version_doc = {
            "base_model": config["base_model"],
            "fine_tuning_config": config["fine_tuning_config"],
            "training_data": config["training_data"],
            "created_at": datetime.utcnow(),
            "status": "initialized",
            "metrics": {},
        }
        result = self.versions_collection.insert_one(version_doc)
        return str(result.inserted_id)

    def update_version_status(self, version_id: str, status: str, metrics: Dict = None):
        """Update model version status and metrics"""
        update_doc = {"status": status, "updated_at": datetime.utcnow()}
        if metrics:
            update_doc["metrics"] = metrics

        self.versions_collection.update_one(
            {"_id": ObjectId(version_id)}, {"$set": update_doc}
        )

    def get_version_info(self, version_id: str) -> Optional[Dict]:
        """Get model version information"""
        return self.versions_collection.find_one({"_id": ObjectId(version_id)})


class LLMTrainer:
    def __init__(self):
        self.model_tracker = ModelTracker()
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=AppConfig.AWS_ACCESS_KEY,
            aws_secret_access_key=AppConfig.AWS_SECRET_KEY,
            endpoint_url=AppConfig.AWS_ENDPOINT_URL,
        )

    def _prepare_training_data(self, config: Dict) -> Dict:
        """Prepare training data from documents"""
        data_sources = config["training_data"]["sources"]
        processed_data = []

        for source in data_sources:
            if source["type"] == "s3":
                # Download and process files from S3
                content = self._get_s3_content(source["path"])
                processed_data.extend(self._process_content(content, source["format"]))

        return {
            "train": processed_data[: int(len(processed_data) * 0.9)],
            "eval": processed_data[int(len(processed_data) * 0.9) :],
        }

    def _get_s3_content(self, path: str) -> str:
        """Get content from S3"""
        try:
            response = self.s3_client.get_object(Bucket=AppConfig.S3_BUCKET, Key=path)
            return response["Body"].read().decode("utf-8")
        except Exception as e:
            logger.error(f"Error reading from S3: {e}")
            raise

    def _process_content(self, content: str, format_type: str) -> List[Dict]:
        """Process content based on format"""
        if format_type == "json":
            data = json.loads(content)
            return self._format_conversations(data)
        elif format_type == "text":
            return self._format_text(content)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _format_conversations(self, data: List[Dict]) -> List[Dict]:
        """Format conversation data for training"""
        formatted = []
        for conv in data:
            formatted.append({"input_ids": conv["input"], "labels": conv["output"]})
        return formatted

    def train(self, config: Dict) -> bool:
        """Train or fine-tune LLM model"""
        version_id = None
        model = None
        trainer = None

        try:
            # Create version and get ID
            version_id = self.model_tracker.create_model_version(config)
            self.model_tracker.update_version_status(version_id, "training")

            # Memory management
            torch.cuda.empty_cache()

            # Validate config
            self._validate_config(config)

            # Load model with error handling
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    config["base_model"],
                    torch_dtype=torch.float16,
                    device_map="auto",
                    use_auth_token=AppConfig.HF_TOKEN,
                )
                tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
            except Exception as e:
                raise ValueError(f"Failed to load model: {e}")

            # Enable gradient checkpointing
            model.gradient_checkpointing_enable()

            # Configure LoRA
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query_key_value"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            # Prepare model for training
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)

            # Prepare training data
            train_data = self._prepare_training_data(config)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=f"./models/{version_id}",
                num_train_epochs=config["fine_tuning_config"]["epochs"],
                per_device_train_batch_size=config["fine_tuning_config"]["batch_size"],
                gradient_accumulation_steps=4,
                learning_rate=config["fine_tuning_config"].get("learning_rate", 2e-4),
                fp16=True,
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="steps",
                eval_steps=100,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
                report_to=["tensorboard"],
                remove_unused_columns=False,
            )

            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_data["train"],
                eval_dataset=train_data["eval"],
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )

            # Train and get results
            train_result = trainer.train()
            metrics = train_result.metrics

            # Save model
            save_path = f"models/{version_id}"
            trainer.save_model(save_path)
            tokenizer.save_pretrained(save_path)

            # Upload to S3
            self._upload_model_to_s3(save_path, version_id)

            # Update status
            self.model_tracker.update_version_status(version_id, "completed", metrics)

            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if version_id:
                self.model_tracker.update_version_status(
                    version_id, "failed", {"error": str(e)}
                )
            return False

        finally:
            # Cleanup
            torch.cuda.empty_cache()
            if model:
                del model
            if trainer:
                del trainer

    def _upload_model_to_s3(self, local_path: str, version_id: str):
        """Upload model files to S3"""
        try:
            s3_path = f"models/llm/{version_id}"
            for file_path in Path(local_path).rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{s3_path}/{relative_path}"
                    self.s3_client.upload_file(
                        str(file_path), AppConfig.S3_BUCKET, s3_key
                    )
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise

    def _validate_config(self, config: Dict):
        """Validate training configuration"""
        required_fields = [
            "base_model",
            "fine_tuning_config.batch_size",
            "fine_tuning_config.epochs",
            "data_source.location",
        ]

        for field in required_fields:
            keys = field.split(".")
            current = config
            for key in keys:
                if not isinstance(current, dict) or key not in current:
                    raise ValueError(f"Missing required field: {field}")
                current = current[key]


class LLMTrainingWorker:
    def __init__(self):
        self.trainer = LLMTrainer()
        self.redis = redis.Redis(
            host=AppConfig.REDIS_HOST,
            port=AppConfig.REDIS_PORT,
            password=AppConfig.REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=5,
            retry_on_timeout=True,
            socket_keepalive=True,
        )
        self.should_stop = False
        self.current_job = None
        self._worker_thread = None
        self.mongo_client = MongoClient(AppConfig.MONGO_URI)
        self.db = self.mongo_client[AppConfig.MONGO_DB]

    def process_job(self, job: Dict):
        """Process a single training job"""
        try:
            config = job.get("config", {})
            config_id = config.get("_id")

            if not config_id:
                raise ValueError("Invalid job structure - missing config ID")

            logger.info(f"Processing job for config: {config_id}")
            self._update_status(config_id, "processing", progress=0)

            # Convert config format if needed
            training_config = {
                "base_model": config.get("base_model"),
                "fine_tuning_config": {
                    "batch_size": config.get("training_config", {}).get(
                        "batch_size", 4
                    ),
                    "epochs": config.get("training_config", {}).get("epochs", 3),
                    "learning_rate": config.get("training_config", {}).get(
                        "learning_rate", 2e-4
                    ),
                },
                "data_source": {
                    "location": config.get("data_source", {}).get("location"),
                    "type": config.get("data_source", {}).get("type"),
                },
            }

            success = self.trainer.train(training_config)

            if success:
                self._update_status(config_id, "completed", progress=100)
            else:
                self._update_status(config_id, "failed", error="Training failed")

        except Exception as e:
            logger.error(f"Error processing job: {e}")
            if config and config_id:
                self._update_status(config_id, "failed", error=str(e))
            else:
                logger.error("Could not update status: invalid job structure")

    def _update_status(
        self, config_id: str, status: str, progress: float = None, error: str = None
    ):
        """Update training status in both Redis and MongoDB"""
        try:
            # Update Redis status
            status_data = {
                "status": status,
                "timestamp": datetime.now().isoformat(),
            }
            if progress is not None:
                status_data["training_stats"] = {
                    "progress": progress,
                    "processed_records": 0,
                    "total_records": 0,
                    "start_time": datetime.now().isoformat() if progress == 0 else None,
                    "end_time": datetime.now().isoformat() if progress == 100 else None,
                }
            if error:
                status_data["error"] = error

            # Update Redis
            key = f"llm_model_status:{config_id}"
            self.redis.set(key, json.dumps(status_data), ex=86400)  # 24 hour expiry

            # Update MongoDB
            update_doc = {"status": status, "updated_at": datetime.now()}
            if progress is not None:
                update_doc["training_stats.progress"] = progress
            if error:
                update_doc["training_stats.error"] = error
            if progress == 0:
                update_doc["training_stats.start_time"] = datetime.now()
            if progress == 100:
                update_doc["training_stats.end_time"] = datetime.now()

            self.db.llm_configs.update_one(
                {"_id": ObjectId(config_id)}, {"$set": update_doc}
            )

        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def _worker_loop(self):
        """Main worker loop consuming from llm-training-queue"""
        logger.info("Starting LLM training worker loop")
        while not self.should_stop:
            try:
                # Get job from llm-training-queue
                result = self.redis.brpop("llm-training-queue", timeout=1)
                if result:
                    _, job_data = result
                    job = json.loads(job_data)
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
            logger.info("LLM training worker started")

    def stop(self):
        """Stop the training worker"""
        logger.info("Stopping LLM training worker")
        self.should_stop = True
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            logger.info("LLM training worker stopped")


# Initialize components
training_worker = LLMTrainingWorker()


@app.route("/train", methods=["POST"])
def train_model():
    """Endpoint to queue a new training job"""
    try:
        config = request.json

        # Validate config
        required_fields = ["base_model", "fine_tuning_config", "training_data"]
        if not all(field in config for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        # Add job to queue
        job = {"config": config}
        job_str = json.dumps(job)
        training_worker.redis.lpush("llm-training-queue", job_str)

        return jsonify({"status": "queued"})

    except Exception as e:
        logger.error(f"Error queueing training job: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/versions/<version_id>")
def get_version(version_id):
    """Get model version information"""
    try:
        tracker = ModelTracker()
        version_info = tracker.get_version_info(version_id)
        if version_info:
            version_info["_id"] = str(version_info["_id"])
            return jsonify(version_info)
        return jsonify({"error": "Version not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "worker_active": training_worker._worker_thread is not None
            and training_worker._worker_thread.is_alive(),
        }
    )


def initialize_service():
    """Initialize the training service"""
    try:
        training_worker.start()
        logger.info("Training service initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        return False


if __name__ == "__main__":
    if initialize_service():
        app.run(host="0.0.0.0", port=AppConfig.SERVICE_PORT)
    else:
        exit(1)

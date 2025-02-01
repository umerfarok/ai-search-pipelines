

import os
from dataclasses import dataclass, field
from typing import Dict

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
    BASE_MODEL_DIR: str = os.getenv("BASE_MODEL_DIR", "/app/models")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")
    TRANSFORMER_CACHE: str = os.getenv("TRANSFORMER_CACHE", "/app/model_cache/transformers")
    HF_HOME: str = os.getenv("HF_HOME", "/app/model_cache/huggingface")

    @classmethod
    def setup_cache_dirs(cls):
        """Setup cache directories for models"""
        for directory in [
            cls.BASE_MODEL_DIR,
            cls.MODEL_CACHE_DIR,
            cls.TRANSFORMER_CACHE,
            cls.HF_HOME,
            os.path.join(cls.HF_HOME, "datasets"),
        ]:
            os.makedirs(directory, exist_ok=True)

        os.environ.update({
            "TORCH_HOME": cls.MODEL_CACHE_DIR,
            "TRANSFORMERS_CACHE": cls.TRANSFORMER_CACHE,
            "HF_HOME": cls.HF_HOME,
            "HF_DATASETS_CACHE": os.path.join(cls.HF_HOME, "datasets"),
        })
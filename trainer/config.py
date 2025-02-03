from dataclasses import dataclass
import os
@dataclass
class AppConfig:
    # Redis Configuration
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    TRAINING_QUEUE: str = "training_queue"
    MODEL_STATUS_PREFIX: str = "model_status:"
    REQUIRED_MODELS = ["BAAI/bge-large-en-v1.5"]
    HF_TOKEN= os.getenv("HF_TOKEN", "asjfnasdjdnflasnfaskdnf")
    
    # Service Configuration
    SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", 5001))
    TRAIN_SERVICE_PORT: int = int(os.getenv("TRAIN_SERVICE_PORT", 5000))
    
    # AWS Configuration
    AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY")
    S3_BUCKET: str = os.getenv("S3_BUCKET")
    S3_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_SSL_VERIFY: str = os.getenv("AWS_SSL_VERIFY", "true")
    AWS_ENDPOINT_URL: str = os.getenv("AWS_ENDPOINT_URL")
    
    # Retry Configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1
    
    # Model Configuration
    DEFAULT_MODEL = "sentence-transformers/all-minilm-l6-v2"  # Update default model
    MODEL_MAPPINGS = {
        "all-minilm-l6": {
            "name": "All-MiniLM-L6",
            "path": "sentence-transformers/all-MiniLM-L6-v2",
            "description": "Fast and efficient general-purpose embedding model",
            "tags": ["semantic", "fast", "default"],
            "is_default": True
        },
        "bge-small": {
            "name": "BGE Small",
            "path": "BAAI/bge-small-en-v1.5",
            "description": "Small but effective embedding model",
            "tags": ["semantic", "fast"],
            "is_default": False
        },
        "bge-base": {
            "name": "BGE Base",
            "path": "BAAI/bge-base-en-v1.5",
            "description": "Medium-sized embedding model",
            "tags": ["semantic", "balanced"],
            "is_default": False
        },
        "bge-large": {
            "name": "BGE Large",
            "path": "BAAI/bge-large-en-v1.5",
            "description": "Large, high-performance embedding model",
            "tags": ["semantic", "accurate"],
            "is_default": False
        }
    }

    # Default model settings
    DEFAULT_MODEL = "all-minilm-l6"
    REQUIRED_MODELS = ["all-minilm-l6", "bge-large"]  # Models to preload
    
    @classmethod
    def get_model_path(cls, model_key: str) -> str:
        """Get the full model path from a model key"""
        if model_key in cls.MODEL_MAPPINGS:
            return cls.MODEL_MAPPINGS[model_key]["path"]
        return model_key  # Return as-is if not found
    
    @classmethod
    def get_model_key(cls, model_path: str) -> str:
        """Get the model key from a full path"""
        for key, info in cls.MODEL_MAPPINGS.items():
            if info["path"] == model_path:
                return key
        return model_path 
    
    # Cache Configuration
    MODEL_CACHE_SIZE: int = int(os.getenv("MODEL_CACHE_SIZE", "3"))
    BASE_MODEL_DIR: str = os.getenv("BASE_MODEL_DIR", "/app/models")
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/app/model_cache")
    TRANSFORMER_CACHE: str = os.getenv("TRANSFORMER_CACHE", "/app/model_cache/transformers")
    HF_HOME: str = os.getenv("HF_HOME", "/app/model_cache/huggingface")
    ONNX_CACHE_DIR: str = os.getenv("ONNX_CACHE_DIR", "/app/model_cache/onnx")
    
    # Search Configuration
    DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Update default model
    MIN_SCORE: float = 0.3
    HYBRID_SEARCH_ALPHA: float = 0.7
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    

    @classmethod
    def setup_cache_dirs(cls):
        """Setup cache directories for models"""
        cache_dirs = [
            cls.BASE_MODEL_DIR,
            cls.MODEL_CACHE_DIR,
            cls.TRANSFORMER_CACHE,
            cls.HF_HOME,
            cls.ONNX_CACHE_DIR,
            os.path.join(cls.HF_HOME, "datasets"),
        ]
        
        for directory in cache_dirs:
            os.makedirs(directory, exist_ok=True)

        # Set environment variables
        os.environ.update({
            "TORCH_HOME": cls.MODEL_CACHE_DIR,
            "TRANSFORMERS_CACHE": cls.TRANSFORMER_CACHE,
            "HF_HOME": cls.HF_HOME,
            "HF_DATASETS_CACHE": os.path.join(cls.HF_HOME, "datasets"),
            "TRANSFORMERS_OFFLINE": "0",  # Allow downloads but will use cache first
            "HF_DATASETS_OFFLINE": "0",   # Allow downloads but will use cache first
            "USE_AUTH_TOKEN": "0",        # Don't require authentication for public models
            "HF_HUB_CACHE": cls.HF_HOME,  # Explicit hub cache location
            "HF_HUB_DISABLE_TELEMETRY": "1"  # Disable telemetry
        })

import os
import logging
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from typing import List, Optional
import threading
import time

logger = logging.getLogger(__name__)

class ModelInitializer:
    def __init__(self):
        self.base_path = Path("/app/models")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.required_models = [
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            # Add other models you might need
        ]
        self.downloaded_models = set()
        self._init_lock = threading.Lock()
        
    def ensure_model_downloaded(self, model_name: str) -> bool:
        """Ensure a specific model is downloaded"""
        try:
            logger.info(f"Checking model: {model_name}")
            model_path = self.base_path / model_name.replace("/", "_")
            
            if model_path in self.downloaded_models:
                logger.info(f"Model {model_name} already downloaded")
                return True
                
            with self._init_lock:
                # Double-check after acquiring lock
                if model_path in self.downloaded_models:
                    return True
                    
                logger.info(f"Downloading model {model_name}...")
                # This will download the model to the specified path
                model = SentenceTransformer(model_name, cache_folder=str(self.base_path))
                model.save(str(model_path))
                self.downloaded_models.add(model_path)
                logger.info(f"Successfully downloaded model {model_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False
            
    def init_models(self) -> bool:
        """Initialize all required models"""
        try:
            # Create base directory
            os.makedirs(self.base_path, exist_ok=True)
            
            # Download all required models
            success = True
            for model_name in self.required_models:
                if not self.ensure_model_downloaded(model_name):
                    success = False
                    
            return success
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            return False
            
    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get the local path for a model"""
        model_path = self.base_path / model_name.replace("/", "_")
        return str(model_path) if model_path in self.downloaded_models else None
# model_optimizer.py
from typing import Dict, Optional
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import json
from dataclasses import dataclass
import os

@dataclass
class ModelCache:
    embeddings: np.ndarray
    products: list
    metadata: dict
    embedding_model: SentenceTransformer
    zero_shot_model: Optional[pipeline] = None

class ModelOptimizer:
    _instance = None
    _cache: Dict[str, ModelCache] = {}
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_model(self, model_path: str) -> ModelCache:
        """Load model with optimizations and caching."""
        absolute_path = str(Path(model_path).resolve())
        
        # Return cached model if available
        if absolute_path in self._cache:
            print(f"Using cached model for {absolute_path}")
            return self._cache[absolute_path]
            
        print(f"Loading model from {absolute_path}")
        
        try:
            # Load metadata first to check compatibility
            metadata = self._load_json(Path(absolute_path) / 'metadata.json')
            
            # Optimize CUDA usage
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                torch.backends.cudnn.benchmark = True
            
            # Load embeddings with memory mapping for large files
            embeddings = np.load(Path(absolute_path) / 'embeddings.npy', mmap_mode='r')
            
            # Load products
            products = self._load_json(Path(absolute_path) / 'products.json')
            
            # Initialize models with optimizations
            embedding_model = self._load_embedding_model(
                absolute_path,
                metadata['config']['training_config']['embedding_model'],
                device
            )
            
            # Initialize zero-shot model only if needed
            zero_shot_model = None
            if metadata['config']['training_config'].get('zero_shot_model'):
                zero_shot_model = self._load_zero_shot_model(device)
            
            # Cache the loaded models
            self._cache[absolute_path] = ModelCache(
                embeddings=embeddings,
                products=products,
                metadata=metadata,
                embedding_model=embedding_model,
                zero_shot_model=zero_shot_model
            )
            
            print(f"Successfully loaded and cached model from {absolute_path}")
            return self._cache[absolute_path]
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def _load_json(self, path: Path) -> dict:
        """Load JSON file with error handling."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON from {path}: {str(e)}")
            raise

    def _load_embedding_model(self, 
                            model_path: str, 
                            model_name: str, 
                            device: str) -> SentenceTransformer:
        """Load embedding model with optimizations."""
        try:
            model = SentenceTransformer(model_name)
            model.to(device)
            
            # Load saved state if available
            state_dict_path = Path(model_path) / 'embedding_model.pt'
            if state_dict_path.exists():
                state_dict = torch.load(state_dict_path, map_location=device)
                model.load_state_dict(state_dict)
            
            if device == 'cuda':
                model.half()  # Use FP16 for CUDA devices
            
            return model
            
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            raise

    def _load_zero_shot_model(self, device: str) -> pipeline:
        """Load zero-shot model with optimizations."""
        try:
            return pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if device == 'cuda' else -1,
                model_kwargs={"torch_dtype": torch.float16 if device == 'cuda' else torch.float32}
            )
        except Exception as e:
            print(f"Error loading zero-shot model: {str(e)}")
            raise

    def clear_cache(self, model_path: Optional[str] = None):
        """Clear model cache for specific or all models."""
        if model_path:
            absolute_path = str(Path(model_path).resolve())
            if absolute_path in self._cache:
                del self._cache[absolute_path]
                print(f"Cleared cache for {absolute_path}")
        else:
            self._cache.clear()
            print("Cleared all model caches")
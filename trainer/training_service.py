# training_service.py
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
from typing import Dict, List, Optional
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class Config:
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    TRAINING_QUEUE = os.getenv('TRAINING_QUEUE', 'training_queue')
    MODEL_PREFIX = os.getenv('MODEL_PREFIX', 'model_status')
    SERVICE_PORT = int(os.getenv('SERVICE_PORT', 5001))
    MODEL_NAME = 'all-mpnet-base-v2'
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 128))
    PROCESSED_FILE_NAME = 'processed_texts.csv'
    API_HOST = os.getenv('API_HOST', 'http://localhost:8080')

def resolve_path(file_path: str) -> str:
    """Convert relative paths to absolute container paths"""
    if file_path.startswith('./'):
        return os.path.join('/app', file_path[2:])
    return file_path

class RedisManager:
    def __init__(self):
        self.redis = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            decode_responses=True
        )
        logger.info(f"Connected to Redis at {Config.REDIS_HOST}:{Config.REDIS_PORT}")

    def get_training_job(self) -> Optional[Dict]:
        try:
            result = self.redis.brpop(Config.TRAINING_QUEUE, timeout=1)
            if result:
                _, job_data = result
                return json.loads(job_data)
        except Exception as e:
            logger.error(f"Error getting training job: {e}")
        return None
    def _resolve_path(self, file_path: str) -> str:
        """Resolve file path"""
        if file_path.startswith('./'):
            return file_path.replace('./', '/app/')
        return file_path

    def update_status(self, version_id: str, status: Dict):
        try:
            key = f"{Config.MODEL_PREFIX}:{version_id}"
            self.redis.set(key, json.dumps(status))
        except Exception as e:
            logger.error(f"Error updating status: {e}")

class ModelTrainer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        self.model = SentenceTransformer(Config.MODEL_NAME)
        self.model.to(self.device)
        self.redis_manager = RedisManager()

   
    def _process_training_data(self, df: pd.DataFrame, config: Dict) -> List[str]:
        """Process training data into text format"""
        schema = config['schema_mapping']
        text_parts = []
        
        # Process main columns
        for column in ['name_column', 'description_column']:
            if schema.get(column) and schema[column] in df.columns:
                text_parts.append(df[schema[column]].fillna('').astype(str))
        
        # Process category
        if schema.get('category_column') and schema['category_column'] in df.columns:
            text_parts.append(df[schema['category_column']].fillna('').astype(str))
        
        # Process custom columns marked for training
        for col in schema.get('custom_columns', []):
            if col['role'] == 'training' and col['user_column'] in df.columns:
                text_parts.append(df[col['user_column']].fillna('').astype(str))
        
        # Combine all text parts
        texts = [' '.join(filter(None, row)) for row in zip(*text_parts)]
        return texts

    def train(self, config: Dict, output_dir: str) -> bool:
        """Main training function"""
        try:
            # Load data
            df = self._load_data(config['data_source']['location'])
            logger.info(f"Loaded {len(df)} records")
            
            # Process training data
            texts = self._process_training_data(df, config)
            logger.info(f"Processed {len(texts)} text samples")
            
            # Get batch size from config or use default
            batch_size = config.get('training_config', {}).get('batch_size', 128)
            
            # Generate embeddings
            embeddings = self._generate_embeddings(texts, batch_size)
            logger.info(f"Generated embeddings of shape {embeddings.shape}")
            
            # Save artifacts
            self._save_artifacts(df, texts, embeddings, config, output_dir)
            logger.info("Training completed successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _process_training_data(self, df: pd.DataFrame, config: Dict) -> List[str]:
        """Process training data into text format"""
        try:
            schema = config['schema_mapping']
            text_parts = []
            
            logger.info(f"Processing data with columns: {df.columns.tolist()}")
            logger.info(f"Using schema mapping: {schema}")
            
            # Process main columns
            for column in ['name_column', 'description_column']:
                if schema.get(column) and schema[column] in df.columns:
                    text_parts.append(df[schema[column]].fillna('').astype(str))
            
            # Process category
            if schema.get('category_column') and schema['category_column'] in df.columns:
                text_parts.append(df[schema['category_column']].fillna('').astype(str))
            
            # Process custom columns marked for training
            for col in schema.get('custom_columns', []):
                if col['role'] == 'training' and col['user_column'] in df.columns:
                    text_parts.append(df[col['user_column']].fillna('').astype(str))
            
            # Combine all text parts
            if not text_parts:
                raise ValueError("No valid columns found for text processing")
                
            texts = [' '.join(filter(None, row)) for row in zip(*text_parts)]
            logger.info(f"Processed {len(texts)} text samples")
            
            # Log a sample for verification
            if texts:
                logger.info(f"Sample processed text: {texts[0][:200]}...")
            
            return texts
            
        except Exception as e:
            logger.error(f"Error processing training data: {e}")
            raise


    def _generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for texts"""
        try:
            # Ensure batch_size is integer
            if isinstance(batch_size, str):
                batch_size = int(batch_size)
            
            logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
            
            if not texts:
                raise ValueError("No texts provided for embedding generation")

            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            embeddings_numpy = embeddings.cpu().numpy()
            logger.info(f"Generated embeddings with shape: {embeddings_numpy.shape}")
            return embeddings_numpy
            
        except ValueError as ve:
            logger.error(f"Value error in embedding generation: {ve}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embedding generation: {e}")
            raise
    def _save_artifacts(self, df: pd.DataFrame, texts: List[str], embeddings: np.ndarray, 
                       config: Dict, output_dir: str) -> None:
        """Save all training artifacts"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save original data
            df.to_csv(output_path / 'products.csv', index=False)
            
            # Save processed texts
            pd.DataFrame({'text': texts}).to_csv(output_path / 'processed_texts.csv', index=False)
            
            # Save embeddings
            np.save(output_path / 'embeddings.npy', embeddings)
            
            # Save model - using self.model here
            self.model.save(str(output_path))
            
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'config': config,
                'num_samples': len(texts),
                'embedding_shape': embeddings.shape,
                'model_name': Config.MODEL_NAME,
                'columns': df.columns.tolist()
            }
            
            with open(output_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Artifacts saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {e}")
            raise e
            
    def _resolve_path(self, file_path: str) -> str:
        """Resolve file path"""
        if file_path.startswith('./'):
            return file_path.replace('./', '/app/')
        return file_path
    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file"""
        try:
            path = self._resolve_path(file_path)
            logger.info(f"Loading data from: {path}")
            return pd.read_csv(path)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def _update_status(self, version_id: str, status: str, 
                      progress: float = None, error: str = None):
        status_data = {
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        if progress is not None:
            status_data["progress"] = progress
        if error:
            status_data["error"] = error

        self.redis_manager.update_status(version_id, status_data)

class TrainingWorker:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.redis_manager = RedisManager()
        self.should_stop = False
        
        
    def _update_model_version(self, version_id: str, status: str, error: str = None):
        """Update model version status in MongoDB"""
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
            
            if error:
                update_data["error"] = error
            
            # Update MongoDB via API
            update_url = f"http://api:8080/model/version/{version_id}/status"
            response = requests.put(
                update_url,
                json=update_data,
                timeout=5
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to update model version status: {response.text}")
        
        except Exception as e:
            logger.error(f"Error updating model version: {e}")

    def start(self):
        logger.info("Starting training worker")
        while not self.should_stop:
            try:
                # Get next job from queue
                job = self.redis_manager.get_training_job()
                if not job:
                    time.sleep(1)
                    continue
                
                version_id = job.get('version_id')
                logger.info(f"Processing job: {version_id}")
                
                if not version_id:
                    logger.error("Job missing version_id")
                    continue

                self.current_job = job
                
                # Extract required parameters from job
                config = job.get('config')
                output_path = job.get('output_path')
                
                if not config or not output_path:
                    raise ValueError(f"Invalid job data: config={bool(config)}, output_path={bool(output_path)}")
                
                # Update status to processing
                self.redis_manager.update_status(version_id, {
                    "status": "processing",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Execute training
                success = self.trainer.train(config=config, output_dir=output_path)
                
                # Update status based on result
                if success:
                    self.redis_manager.update_status(version_id, {
                        "status": "completed",
                        "timestamp": datetime.now().isoformat()
                    })
                    self._update_model_version(version_id, "completed",)
                    logger.info(f"Training completed for job: {version_id}")
                else:
                    self.redis_manager.update_status(version_id, {
                        "status": "failed",
                        "timestamp": datetime.now().isoformat()
                    })
                    self._update_model_version(version_id, "failed", "Training failed")
                    raise Exception("Training returned False")

                
                self.current_job = None
                
            except Exception as e:
                logger.error(f"Error processing job: {e}")
                if self.current_job and self.current_job.get('version_id'):
                    self.redis_manager.update_status(self.current_job['version_id'], {
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                time.sleep(1)

    def stop(self):
        logger.info("Stopping training worker")
        self.should_stop = True

# Initialize worker
worker = TrainingWorker()
worker_thread = threading.Thread(target=worker.start)
worker_thread.daemon = True
worker_thread.start()

@app.route('/health')
def health():
    try:
        # Check if model directories are accessible
        model_dirs = Path('/app/models').glob('*')
        
        return jsonify({
            "status": "healthy",
            "models_available": True
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/status/<version_id>')
def get_status(version_id):
    status = worker.redis_manager.redis.get(
        f"{Config.MODEL_PREFIX}:{version_id}"
    )
    return jsonify(json.loads(status) if status else {"status": "not_found"})

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('/app/data/products', exist_ok=True)
    os.makedirs('/app/models', exist_ok=True)
    
    app.run(host='0.0.0.0', port=Config.SERVICE_PORT)
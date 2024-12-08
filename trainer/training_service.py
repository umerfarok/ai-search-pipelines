# training_service.py
from flask import Flask, request, jsonify
import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import logging
from pathlib import Path
import threading
import queue
from query_patterns import PRODUCT_CATEGORIES, get_category_keywords

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)

# Global settings
MODEL_NAME = 'all-mpnet-base-v2'
BATCH_SIZE = 32

class TrainingData:
    def __init__(self, df: pd.DataFrame, texts: list, embeddings: np.ndarray):
        self.df = df
        self.texts = texts
        self.embeddings = embeddings

class DataProcessor:
    @staticmethod
    def resolve_path(file_path: str) -> Path:
        if file_path.startswith('./'):
            return Path('/app') / file_path[2:]
        return Path(file_path)

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        path = DataProcessor.resolve_path(file_path)
        logger.info(f"Loading CSV from resolved path: {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        try:
            return pd.read_csv(path)
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise

class TrainingManager:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        self.embedding_model = SentenceTransformer(MODEL_NAME)
        self.embedding_model.to(self.device)

    def process_training_data(self, df: pd.DataFrame, config: dict) -> tuple:
        schema = config['schema_mapping']
        text_parts = []
        
        # Process main columns
        for column in ['name_column', 'description_column']:
            if schema.get(column) and schema[column] in df.columns:
                text_parts.append(df[schema[column]].fillna('').astype(str))
        
        # Add category information
        if schema.get('category_column') and schema['category_column'] in df.columns:
            categories = df[schema['category_column']].fillna('').astype(str)
            category_texts = []
            
            for cat in categories:
                keywords = []
                for category, data in PRODUCT_CATEGORIES.items():
                    if any(keyword in cat.lower() for keyword in data['keywords']):
                        keywords.extend(data['keywords'])
                category_texts.append(' '.join(set(keywords)))
            
            text_parts.append(pd.Series(category_texts))
        
        # Process custom columns
        for col in schema.get('custom_columns', []):
            if col['role'] == 'training' and col['user_column'] in df.columns:
                text_parts.append(df[col['user_column']].fillna('').astype(str))
        
        # Combine text parts
        texts = [' '.join(row) for row in zip(*text_parts)]
        logger.info(f"Processed {len(texts)} training samples")
        
        return texts

    def generate_embeddings(self, texts: list) -> np.ndarray:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            
            embeddings = embeddings.cpu().numpy()
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def save_artifacts(self, training_data: TrainingData, config: dict, output_dir: str):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving artifacts to {output_path}")

        try:
            # Save embeddings
            np.save(output_path / 'embeddings.npy', training_data.embeddings)
            
            # Save model
            torch.save(
                self.embedding_model.state_dict(),
                output_path / 'embedding_model.pt'
            )
            
            # Prepare and save products
            schema = config['schema_mapping']
            products = []
            
            for idx, row in training_data.df.iterrows():
                product = {
                    'id': str(row.get(schema['id_column'], idx)),
                    'name': str(row.get(schema['name_column'], '')),
                    'description': str(row.get(schema['description_column'], '')),
                    'category': str(row.get(schema.get('category_column', ''), '')),
                    'metadata': {}
                }
                
                # Add metadata from custom columns
                for col in schema.get('custom_columns', []):
                    if col['role'] == 'metadata' and col['user_column'] in training_data.df.columns:
                        product['metadata'][col['standard_column']] = str(row[col['user_column']])
                
                products.append(product)
            
            # Save products
            with open(output_path / 'products.json', 'w', encoding='utf-8') as f:
                json.dump(products, f, indent=2, ensure_ascii=False)
            
            # Save metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'config': config,
                'embedding_shape': training_data.embeddings.shape,
                'model_name': MODEL_NAME,
                'num_products': len(products),
                'device_used': self.device
            }
            
            with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved all artifacts to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {e}")
            raise

    def train(self, config: dict, output_dir: str):
        try:
            # Load data
            df = DataProcessor.load_csv(config['data_source']['location'])
            
            # Process data
            texts = self.process_training_data(df, config)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Create training data object
            training_data = TrainingData(df, texts, embeddings)
            
            # Save artifacts
            self.save_artifacts(training_data, config, output_dir)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

# Training queue and worker
training_queue = queue.Queue()
active_trainings = {}

def training_worker():
    while True:
        try:
            training_id, config, output_dir = training_queue.get()
            logger.info(f"Processing training job {training_id}")
            
            manager = TrainingManager()
            manager.train(config, output_dir)
            
            active_trainings[training_id] = {'status': 'completed'}
            logger.info(f"Training {training_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            active_trainings[training_id] = {
                'status': 'failed',
                'error': str(e)
            }
        finally:
            training_queue.task_done()

# Start worker thread
threading.Thread(target=training_worker, daemon=True).start()

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        training_id = data['training_id']
        config = data['config']
        output_dir = data['output_dir']
        
        # Ensure output directory is absolute
        output_dir = str(DataProcessor.resolve_path(output_dir))
        
        training_queue.put((training_id, config, output_dir))
        active_trainings[training_id] = {'status': 'queued'}
        
        return jsonify({
            'training_id': training_id,
            'status': 'queued'
        })
        
    except Exception as e:
        logger.error(f"Error handling train request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/training-status/<training_id>', methods=['GET'])
def training_status(training_id):
    status = active_trainings.get(training_id, {'status': 'not_found'})
    return jsonify(status)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('TRAINING_SERVICE_PORT', 5001))
    app.run(host='0.0.0.0', port=port)
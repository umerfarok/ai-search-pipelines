# search_service.py
import os
import logging
from flask import Flask, request, jsonify
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import redis
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class Config:
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    MODEL_CACHE_SIZE = 2
    MIN_SCORE = 0.2

class ModelManager:
    def __init__(self):
        self.model_cache = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

    def load_model(self, model_path: str) -> Optional[Dict]:
        try:
            model_path = self._resolve_path(model_path)
            
            if model_path in self.model_cache:
                return self.model_cache[model_path]

            logger.info(f"Loading model from {model_path}")

            # Load metadata
            with open(Path(model_path) / 'metadata.json', 'r') as f:
                metadata = json.load(f)

            # Initialize model
            model = SentenceTransformer(metadata['model_name'])
            model.to(self.device)

            # Load saved weights
            model_file = Path(model_path) / 'model.pt'
            if model_file.exists():
                state_dict = torch.load(model_file, map_location=self.device)
                model.load_state_dict(state_dict)

            # Load embeddings
            embeddings = np.load(Path(model_path) / 'embeddings.npy')

            # Cache model data
            model_data = {
                'model': model,
                'embeddings': embeddings,
                'metadata': metadata,
                'last_used': torch.cuda.current_stream().record_event() if torch.cuda.is_available() else None
            }

            # Manage cache size
            if len(self.model_cache) >= Config.MODEL_CACHE_SIZE:
                oldest_path = min(self.model_cache.keys(), key=lambda k: id(self.model_cache[k]['last_used']))
                del self.model_cache[oldest_path]

            self.model_cache[model_path] = model_data
            return model_data

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def _resolve_path(self, path: str) -> str:
        if path.startswith('./'):
            return path.replace('./', '/app/')
        return path

class SearchProcessor:
    def __init__(self):
        self.model_manager = ModelManager()

    def search(self, model_path: str, query: str, max_items: int = 5) -> Dict:
        try:
            # Load model
            model_data = self.model_manager.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            model = model_data['model']
            embeddings = model_data['embeddings']
            metadata = model_data['metadata']

            # Generate query embedding
            query_embedding = model.encode(
                query,
                convert_to_tensor=True,
                normalize_embeddings=True
            ).cpu().numpy()

            # Calculate similarities
            similarities = np.dot(query_embedding, embeddings.T)
            top_k_idx = np.argsort(similarities)[-max_items:][::-1]

            # Load product data
            try:
                import pandas as pd
                products_file = Path(model_path) / 'products.csv'
                if not products_file.exists():
                    raise FileNotFoundError(f"Products file not found at {products_file}")
                
                products_df = pd.read_csv(products_file)
                
                # Log available columns for debugging
                logger.info(f"Available columns in CSV: {products_df.columns.tolist()}")
                logger.info(f"Schema mapping: {metadata['config']['schema_mapping']}")
                
                # Verify schema mapping
                schema = metadata['config']['schema_mapping']
                required_columns = {
                    'id_column': schema.get('id_column'),
                    'name_column': schema.get('name_column'),
                    'description_column': schema.get('description_column'),
                    'category_column': schema.get('category_column')
                }

                # Check for missing columns
                missing_columns = [
                    col_name for col_name, col_value in required_columns.items()
                    if col_value and col_value not in products_df.columns
                ]

                if missing_columns:
                    raise ValueError(f"Missing required columns in CSV: {missing_columns}")

            except Exception as e:
                logger.error(f"Error reading or validating products file: {e}")
                raise

            # Prepare results
            results = []
            for idx in top_k_idx:
                score = float(similarities[idx])
                if score < Config.MIN_SCORE:
                    continue

                try:
                    product = products_df.iloc[idx]
                    
                    # Safe get values with fallbacks
                    result = {
                        "id": str(product.get(schema['id_column'], f"item_{idx}")),
                        "name": str(product.get(schema['name_column'], "Unknown")),
                        "description": str(product.get(schema['description_column'], "")),
                        "category": str(product.get(schema['category_column'], "Uncategorized")),
                        "score": score,
                        "metadata": {}
                    }

                    # Add metadata from custom columns
                    for col in schema.get('custom_columns', []):
                        if col['role'] == 'metadata' and col['user_column'] in product:
                            try:
                                result['metadata'][col['standard_column']] = str(product[col['user_column']])
                            except Exception as e:
                                logger.warning(f"Error processing custom column {col['user_column']}: {e}")

                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing result at index {idx}: {e}")
                    logger.error(f"Product data at index: {products_df.iloc[idx].to_dict()}")
                    continue

            return {
                "results": results,
                "total": len(results),
                "query_info": {
                    "original": query,
                    "model_version": metadata.get('version', 'unknown'),
                    "schema_mapping": schema
                }
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

# Initialize search processor
search_processor = SearchProcessor()

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data or 'model_path' not in data or 'query' not in data:
            return jsonify({"error": "Invalid request"}), 400

        model_path = data['model_path']
        query = data['query']
        max_items = data.get('max_items', 5)

        # Add debug logging
        logger.info(f"Processing search request for model: {model_path}")
        logger.info(f"Query: {query}")

        results = search_processor.search(model_path, query, max_items)
        return jsonify(results)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Search error: {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    port = int(os.getenv('SERVICE_PORT', 5000))
    app.run(host='0.0.0.0', port=port)
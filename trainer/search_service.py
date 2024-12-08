# search_service.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
import torch
import logging
from typing import Dict, List, Optional
import os
import re
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@dataclass
class SearchConfig:
    MODEL_NAME: str = 'all-mpnet-base-v2'
    MIN_SCORE: float = 0.2
    MAX_CACHE_SIZE: int = 2
    BATCH_SIZE: int = 32

class QueryProcessor:
    """Handles query analysis and enhancement"""
    
    PRODUCT_CATEGORIES = {
        'appliance': ['refrigerator', 'fridge', 'freezer', 'washer', 'dryer', 'dishwasher'],
        'feature': ['energy star', 'smart', 'efficient', 'capacity', 'size'],
        'brand': ['lg', 'samsung', 'whirlpool', 'ge', 'maytag'],
        'price': ['cheap', 'expensive', 'cost', 'price', 'affordable', 'budget'],
    }
    
    @staticmethod
    def analyze_query(query: str) -> Dict:
        """Analyze query for intent and features"""
        query_lower = query.lower()
        
        # Extract categories
        categories = []
        keywords = []
        
        for category, terms in QueryProcessor.PRODUCT_CATEGORIES.items():
            if any(term in query_lower for term in terms):
                categories.append(category)
                keywords.extend([term for term in terms if term in query_lower])
        
        # Extract measurements
        measurements = re.findall(r'\d+(?:\.\d+)?\s*(?:feet|ft|inches|in|cubic)', query_lower)
        
        # Detect price ranges
        price_range = re.findall(r'under\s*\$?\s*(\d+)', query_lower)
        
        return {
            'categories': categories,
            'keywords': list(set(keywords)),
            'measurements': measurements,
            'price_range': price_range[0] if price_range else None
        }
    
    @staticmethod
    def enhance_query(query: str, features: Dict) -> str:
        """Enhance query with extracted features"""
        enhanced_parts = [query]
        
        # Add relevant keywords
        if features['keywords']:
            enhanced_parts.extend(features['keywords'])
            
        # Add measurement context
        if features['measurements']:
            enhanced_parts.extend(features['measurements'])
            
        return ' '.join(enhanced_parts)

class ModelInstance:
    def __init__(self, model_path: str):
        self.model_path = self._resolve_path(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_used = datetime.now()
        logger.info(f"Initializing model instance using {self.device}")
        self.load_artifacts()
    
    def _resolve_path(self, path: str) -> Path:
        if path.startswith('./'):
            return Path('/app') / path[2:]
        return Path(path)
    
    def load_artifacts(self):
        try:
            # Load model
            self.embedding_model = SentenceTransformer(SearchConfig.MODEL_NAME)
            self.embedding_model.to(self.device)
            
            # Load saved weights if available
            model_weights = self.model_path / 'embedding_model.pt'
            if model_weights.exists():
                state_dict = torch.load(model_weights, map_location=self.device)
                self.embedding_model.load_state_dict(state_dict)
            
            # Load embeddings
            self.embeddings = np.load(self.model_path / 'embeddings.npy')
            
            # Load metadata
            with open(self.model_path / 'metadata.json', 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
                
            # Load products
            with open(self.model_path / 'products.json', 'r', encoding='utf-8') as f:
                self.products = json.load(f)
                
            logger.info(f"Loaded artifacts: {len(self.products)} products, embeddings shape: {self.embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise

    def search(self, query: str, max_results: int = 5) -> Dict:
        try:
            self.last_used = datetime.now()
            
            # Handle greetings
            if self._is_greeting(query):
                return self._generate_greeting_response()
            
            # Process query
            features = QueryProcessor.analyze_query(query)
            enhanced_query = QueryProcessor.enhance_query(query, features)
            logger.info(f"Query features: {features}")
            
            # Generate embedding
            query_embedding = self.embedding_model.encode(
                enhanced_query,
                convert_to_tensor=True,
                normalize_embeddings=True
            ).cpu().numpy()
            
            # Reshape if needed
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Calculate similarities
            similarities = np.dot(query_embedding, self.embeddings.T).squeeze()
            
            # Get top results
            top_k_idx = np.argsort(similarities)[-max_results:][::-1]
            
            results = []
            for idx in top_k_idx:
                score = float(similarities[idx])
                if score < SearchConfig.MIN_SCORE:
                    continue
                
                product = self.products[idx]
                result = {
                    "id": product["id"],
                    "name": product["name"],
                    "description": product["description"],
                    "category": product["category"],
                    "score": score,
                    "metadata": product.get("metadata", {})
                }
                
                # Add price match info if relevant
                if features['price_range'] and 'price' in result['metadata']:
                    try:
                        price = float(result['metadata']['price'])
                        result['price_match'] = price <= float(features['price_range'])
                    except ValueError:
                        pass
                
                results.append(result)
            
            response = self._generate_response(query, results, features)
            
            return {
                "results": results,
                "total": len(results),
                "query_info": {
                    "original": query,
                    "enhanced": enhanced_query,
                    "features": features
                },
                "response": response
            }
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    def _is_greeting(self, query: str) -> bool:
        greetings = ['hello', 'hi', 'hey', 'greetings']
        query_words = query.lower().split()
        return len(query_words) <= 2 and any(word in greetings for word in query_words)

    def _generate_greeting_response(self) -> Dict:
        return {
            "results": [],
            "total": 0,
            "response": (
                "Hello! I can help you find appliances based on your needs. "
                "You can ask about specific features, brands, prices, or tell me what you're looking for!"
            )
        }

    def _generate_response(self, query: str, results: List[Dict], features: Dict) -> str:
        if not results:
            return (
                "I couldn't find any products matching your requirements. "
                "Could you please try describing what you're looking for differently?"
            )
        
        top_product = results[0]
        
        # Price-focused response
        if features['price_range']:
            price = top_product['metadata'].get('price', 'N/A')
            return (
                f"I found a {top_product['name']} priced at ${price}. "
                f"{top_product['description']}"
            )
        
        # Feature-focused response
        if 'feature' in features['categories']:
            features_list = ', '.join(features['keywords'])
            return (
                f"I found a {top_product['name']} with the features you're looking for "
                f"({features_list}). {top_product['description']}"
            )
        
        # Brand-focused response
        if 'brand' in features['categories']:
            brand = top_product['metadata'].get('brand', 'the manufacturer')
            return (
                f"I found this {brand} {top_product['name']}. "
                f"{top_product['description']}"
            )
        
        # Default response
        return (
            f"Based on your search, I recommend the {top_product['name']}. "
            f"{top_product['description']}"
        )

# Global model cache
model_cache = {}

def get_model_instance(model_path: str) -> ModelInstance:
    if model_path not in model_cache:
        if len(model_cache) >= SearchConfig.MAX_CACHE_SIZE:
            # Remove oldest model from cache
            oldest_path = min(
                model_cache.keys(),
                key=lambda k: model_cache[k].last_used
            )
            del model_cache[oldest_path]
            logger.info(f"Removed {oldest_path} from cache")
        
        model_cache[model_path] = ModelInstance(model_path)
    
    return model_cache[model_path]

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        
        # Validate request
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required_fields = ['model_path', 'query']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400
        
        model_path = data['model_path']
        query = data['query']
        max_items = data.get('max_items', 5)
        
        logger.info(f"Search request - Query: {query}")
        
        model = get_model_instance(model_path)
        results = model.search(query, max_items)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        return jsonify({
            "status": "healthy",
            "cached_models": len(model_cache),
            "max_cache_size": SearchConfig.MAX_CACHE_SIZE
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('SEARCH_SERVICE_PORT', 5000))
    app.run(host='0.0.0.0', port=port)
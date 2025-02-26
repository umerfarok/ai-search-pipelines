import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from flask import Flask, request, jsonify
import requests
import re
from threading import Thread
from queue import Queue
import contextlib
import torch.cuda
from cachetools import TTLCache
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
import json
import time
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy

from vector_store import VectorStore
from simple_cache import TimedCache
from config import AppConfig

# Change logging level for better focus on important messages
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)

# Define common product categories for classification
PRODUCT_CATEGORIES = [
    "electronics", "home appliances", "furniture", "kitchen", "cleaning", 
    "pest control", "pet supplies", "clothing", "toys", "tools", "office supplies",
    "health", "beauty", "food", "sports", "automotive", "garden"
]

# Add product-specific keywords to help with intent classification
PRODUCT_KEYWORDS = {
    "mouse trap": ["mouse", "mice", "trap", "rodent", "pest", "kill", "catch", "bait"],
    "fabric shaver": ["fabric", "lint", "fuzz", "clothes", "remover", "shaver"],
    "cleaning supplies": ["clean", "cleaner", "cleaning", "wash", "mop", "vacuum"],
    "pest control": ["pest", "insect", "bug", "rodent", "spray", "repellent", "kill"]
}

class ProductCategoryClassifier:
    """Lightweight product category classifier to improve search relevance"""
    
    def __init__(self):
        # Initialize with product categories and train basic models
        self.categories = PRODUCT_CATEGORIES
        self.keyword_mapping = PRODUCT_KEYWORDS
        self.vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
        
        # Train with basic category descriptions
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize lightweight classification models"""
        try:
            # Create basic training data from categories and keywords
            training_texts = []
            training_labels = []
            
            for category in self.categories:
                # Add multiple examples per category
                training_texts.append(f"I'm looking for {category} products")
                training_labels.append(category)
                training_texts.append(f"Show me {category}")
                training_labels.append(category)
                
            for product, keywords in self.keyword_mapping.items():
                for keyword in keywords:
                    training_texts.append(f"I need {keyword}")
                    training_labels.append(product)
                    
            # Fit vectorizer on training data
            self.vectorizer.fit(training_texts)
            
            # We'll use pre-trained embedding model for few-shot classification
            # rather than training a full classifier
            
            logger.info(f"Category classifier initialized with {len(self.categories)} categories")
        except Exception as e:
            logger.error(f"Failed to initialize category classifier: {e}")
            
    def classify_query(self, query: str) -> Tuple[str, float]:
        """Identify most likely product category for a query"""
        # First check for direct keyword matches (most precise)
        for product, keywords in self.keyword_mapping.items():
            if any(keyword.lower() in query.lower() for keyword in keywords):
                # Calculate confidence based on number of keyword matches
                matched = sum(1 for k in keywords if k.lower() in query.lower())
                confidence = min(0.95, matched / len(keywords) + 0.7)  # Base confidence + matches
                return product, confidence
        
        # Fall back to general category
        # In a real system, we would use a proper trained classifier here
        query_lower = query.lower()
        
        for category in self.categories:
            if category.lower() in query_lower:
                return category, 0.85
                
        # No direct match, return pest control for "kill mouse" queries
        if "mouse" in query_lower and ("kill" in query_lower or "trap" in query_lower or "catch" in query_lower):
            return "pest control", 0.9
            
        # Return most common category with low confidence
        return "general", 0.3


class OptimizedEmbeddingManager:
    """Optimized embedding manager with better threading and memory management"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_models = {}
        self.model_lock = threading.RLock()  # Use reentrant lock for nested access
        self.cache_dir = AppConfig.TRANSFORMER_CACHE
        self.embedding_dimension = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            # Add more models and their dimensions
        }
        self._local_faiss_index = {}  # Store local FAISS indexes for fast searching
        
    def create_faiss_index(self, model_name: str, embeddings: np.ndarray) -> Any:
        """Create a FAISS index for fast approximate nearest neighbor search"""
        dim = embeddings.shape[1]
        
        # Use an appropriate index type based on dimensionality and data size
        if embeddings.shape[0] < 1000:
            # Small dataset: exact search is fast enough
            index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
        else:
            # Larger dataset: use approximate search
            # M=16 is a good default for HNSW (higher=more accuracy but slower)
            index = faiss.IndexHNSWFlat(dim, 16)
            
        # Convert to float32 if needed
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        # Normalize the vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add vectors to index
        index.add(embeddings)
        
        return index
        
    def get_model(self, model_path: str):
        """Get or load an embedding model with optimized resource handling"""
        # Normalize model path
        if not model_path or model_path == "default":
            model_path = "sentence-transformers/all-MiniLM-L6-v2"
            
        # Check if model is already loaded
        if model_path in self.embedding_models:
            return self.embedding_models[model_path]
            
        # Load model with resource optimization
        try:
            with self.model_lock:
                if model_path not in self.embedding_models:
                    start_time = time.time()
                    
                    # Load model with optimized configuration
                    model = SentenceTransformer(
                        model_path,
                        cache_folder=self.cache_dir,
                        device=self.device
                    )
                    
                    # Apply half-precision for GPU memory optimization
                    if torch.cuda.is_available():
                        model.half()  # Use FP16 for better efficiency
                    
                    self.embedding_models[model_path] = model
                    
                    logger.info(f"Loaded model {model_path} in {time.time() - start_time:.2f}s")
                    
                return self.embedding_models[model_path]
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            # Fall back to default model if available
            if (model_path != "sentence-transformers/all-MiniLM-L6-v2"):
                logger.warning(f"Falling back to default model")
                return self.get_model("sentence-transformers/all-MiniLM-L6-v2")
            raise
                
    def generate_embedding(self, 
                           texts: List[str], 
                           model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
                           batch_size: int = None) -> np.ndarray:
        """Generate embeddings with optimized batch size and parallel processing"""
        if not texts:
            raise ValueError("No texts provided for embedding generation")
            
        # Auto-adjust batch size based on available memory and text length
        if batch_size is None:
            avg_length = sum(len(t.split()) for t in texts) / len(texts)
            # Adjust batch size based on text length and available memory
            if torch.cuda.is_available():
                if avg_length > 100:
                    batch_size = 16
                elif avg_length > 50:
                    batch_size = 32
                else:
                    batch_size = 64
            else:
                # CPU processing - smaller batches
                if avg_length > 100:
                    batch_size = 8
                else:
                    batch_size = 16

        model = self.get_model(model_path)
        
        try:
            start_time = time.time()
            
            # Process in optimized batches
            with torch.no_grad():
                embeddings = model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
                
                # Convert to numpy and ensure correct type
                embeddings_np = embeddings.cpu().numpy()
                
            logger.info(f"Generated {len(texts)} embeddings in {time.time() - start_time:.2f}s")
            return embeddings_np
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class SimpleLLMManager:
    """Simplified LLM manager that works with minimal models"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use a more widely compatible model
        self.model_name = "gpt2"  # Smaller and more compatible
        self.model = None
        self.tokenizer = None
        self.model_lock = threading.Lock()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize a simple model without advanced features"""
        try:
            logger.info(f"Loading simple model: {self.model_name}")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load a simple model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32  # Use FP32 for compatibility
            ).to(self.device)
            
            logger.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise

    @contextlib.contextmanager
    def get_model(self):
        """Context manager to safely access the model"""
        with self.model_lock:
            yield self.model, self.tokenizer

    def generate_response(self, prompt: str, max_length: int = 150) -> str:
        """Generate text response with correct parameters to avoid warnings"""
        try:
            with self.get_model() as (model, tokenizer):
                # Encode prompt
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Generate with properly aligned parameters
                with torch.no_grad():
                    output_sequences = model.generate(
                        **inputs,
                        max_length=len(inputs["input_ids"][0]) + max_length,
                        do_sample=True,  # Set to True to avoid warnings with temperature/top_p
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode output
                generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
                
                # Remove the prompt from the output if needed
                if prompt in generated_text:
                    response = generated_text[len(prompt):].strip()
                else:
                    response = generated_text.strip()
                    
                return response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Unable to provide product recommendations at this time."


class OptimizedSearchService:
    """Improved search service with better performance and relevance"""
    
    def __init__(self):
        # Initialize optimized components
        self.embedding_manager = OptimizedEmbeddingManager()
        self.vector_store = VectorStore()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to initialize the fast LLM first
        try:
            self.llm_manager = SimpleLLMManager()
            logger.info("Using SimpleLLMManager")
        except Exception as e:
            logger.warning(f"Failed to initialize LLM Manager: {e}")
            self.llm_manager = None
        
        # Add category classifier for better intent matching
        self.category_classifier = ProductCategoryClassifier()
        
        # Initialize caches
        self.response_cache = TTLCache(maxsize=2000, ttl=3600)
        self.embedding_cache = TTLCache(maxsize=10000, ttl=3600)
        self.config_cache = TTLCache(maxsize=100, ttl=300)
        
        # For text matching
        self.tokenizer = CountVectorizer(ngram_range=(1, 3))
        
        # Add cross-encoder for reranking (initialize later)
        self.cross_encoder = None
        self._initialize_reranker()

    def _initialize_reranker(self):
        """Initialize cross-encoder for reranking results"""
        try:
            # Note: ms-marco-MiniLM is faster than TAS-B models while still effective
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Cross-encoder reranker initialized")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")

    def _compute_text_similarity(self, query: str, docs: List[str]) -> List[float]:
        """Compute BM25-style text similarity for hybrid search"""
        try:
            # Simple vocabulary-based similarity
            query_tokens = set(query.lower().split())
            scores = []
                    
            for doc in docs:
                doc_tokens = set(doc.lower().split())
                if not doc_tokens:
                    scores.append(0.0)
                    continue
                    
                # Count matching tokens
                matches = len(query_tokens.intersection(doc_tokens))
                score = matches / max(len(query_tokens), 1)
                scores.append(score)
                
            return scores
        except Exception as e:
            logger.error(f"Text similarity calculation failed: {e}")
            return [0.0] * len(docs)

    def _fetch_config(self, model_path: str) -> Dict:
        """Get configuration with improved caching, fallbacks, and debug logging"""
        # Check cache first:
        if model_path in self.config_cache:
            return self.config_cache[model_path]

        try:
            # Extract config ID
            config_id = model_path.split("/")[-1]
            logger.info(f"Fetching config for ID: {config_id}")

            # Try API with improved error handling
            url = f"{AppConfig.API_HOST}/config/{config_id}"
            try:
                logger.info(f"Requesting config from: {url}")
                response = requests.get(url, timeout=5)
                
                # Debug response
                if response.status_code != 200:
                    logger.warning(f"API response status: {response.status_code}, Content: {response.text[:200]}")
                
                response.raise_for_status()
                config = response.json()
                logger.info(f"Successfully fetched config from API: {config.get('id')}")
                self.config_cache[model_path] = config
                return config
            except requests.RequestException as e:
                logger.warning(f"API config fetch failed: {e}")
                logger.info(f"Falling back to collection metadata for {config_id}")

                # Try collection metadata directly (faster fallback)
                collection_name = f"products_{config_id}"
                collection_meta = self.vector_store.get_collection_metadata(collection_name)
                
                if collection_meta:
                    logger.info(f"Using collection metadata for {collection_name}")
                    inferred_config = {
                        "id": config_id,
                        "training_config": {
                            "embeddingmodel": collection_meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
                        },
                        "schema_mapping": collection_meta.get("schema_mapping", {})
                    }
                    self.config_cache[model_path] = inferred_config
                    return inferred_config
                else:
                    logger.warning(f"No metadata found for collection {collection_name}")

            # Last resort: Try to check if the collection exists and create a dummy config
            collection_name = f"products_{config_id}"
            if self.vector_store.collection_exists(collection_name):
                logger.info(f"Collection {collection_name} exists, creating minimal config")
                default_config = {
                    "id": config_id,
                    "name": f"Model {config_id}",
                    "description": "Automatically detected model",
                    "training_config": {
                        "embeddingmodel": "sentence-transformers/all-MiniLM-L6-v2"
                    },
                    "schema_mapping": {
                        "customcolumns": []
                    }
                }
                self.config_cache[model_path] = default_config
                return default_config

            logger.warning(f"Using fallback config for {config_id}")
            # Absolute last resort default config
            default_config = {
                "id": config_id,
                "training_config": {
                    "embeddingmodel": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "schema_mapping": {
                    "customcolumns": []
                }
            }
            self.config_cache[model_path] = default_config
            return default_config

        except Exception as e:
            logger.error(f"Config fetch error: {e}")
            # Minimal fallback config
            return {
                "id": model_path.split("/")[-1],
                "training_config": {"embeddingmodel": "sentence-transformers/all-MiniLM-L6-v2"},
                "schema_mapping": {"customcolumns": []}
            }

    def _detect_intent(self, query: str) -> Dict[str, Any]:
        """Extract search intent from query for better targeting"""
        intent_data = {
            "category": None,
            "confidence": 0.0,
            "is_question": '?' in query or query.lower().startswith(('how', 'what', 'where', 'do you', 'can i', 'is there')),
            "keywords": []
        }
        
        # Extract category and confidence
        category, confidence = self.category_classifier.classify_query(query)
        intent_data["category"] = category
        intent_data["confidence"] = confidence
        
        # Extract key terms as potential filters
        words = query.lower().split()
        intent_data["keywords"] = [w for w in words if len(w) > 3 and w not in 
                                  ['have', 'what', 'where', 'there', 'that', 'with', 'this']]
        
        return intent_data

    def _optimize_query(self, query: str, intent: Dict[str, Any]) -> str:
        """Optimize the query based on detected intent"""
        # For high confidence category matches, explicitly include the category
        if intent["confidence"] > 0.7 and intent["category"] != "general":
            if intent["category"] not in query.lower():
                optimized = f"{query} {intent['category']}"
                logger.info(f"Optimized query: {query} -> {optimized}")
                return optimized

        # For specific product searches, be more direct
        if "mouse" in query.lower() and "kill" in query.lower():
            return "mouse trap rodent pest control"
            
        return query

    def search(self, query: str, model_path: str, top_k: int = 10, 
               filters: Dict = None) -> Dict:
        """Enhanced search with parallel processing and better relevance"""
        start_time = time.time()
        
        try:
            # Clean and normalize query
            query = query.strip().lower()
            
            # Cache check - if we've seen this exact query before
            cache_key = f"{model_path}:{query}"
            if cache_key in self.response_cache:
                logger.info(f"Cache hit for query: {query}")
                return self.response_cache[cache_key]

            # Get collection name
            config_id = model_path.split("/")[-1]
            collection_name = f"products_{config_id}"
            
            # Check if collection exists
            if not self.vector_store.collection_exists(collection_name):
                logger.error(f"Collection {collection_name} not found")
                return {
                    "error": f"Model not found: {model_path}. Please verify the model ID and ensure data is loaded.",
                    "status": "failed",
                    "debug_info": {"collection_name": collection_name}
                }

            # 1. First stage: Detect intent & optimize query
            intent = self._detect_intent(query)
            logger.info(f"Detected intent: {intent}")
            
            # If this is a question about killing mice, direct to pest control
            if intent["category"] == "pest control" and ("mouse" in query or "mice" in query):
                search_query = "mouse trap rodent control"
                logger.info(f"Redirecting to pest control search: {search_query}")
            else:
                search_query = self._optimize_query(query, intent)

            # 2. Fetch relevant data using parallel processing
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Parallel execution of data fetching tasks
                config_future = executor.submit(self._fetch_config, model_path)
                metadata_future = executor.submit(
                    self.vector_store.get_collection_metadata, collection_name
                )
                
                # Get results from futures
                config = config_future.result()
                collection_meta = metadata_future.result()
                
                if not collection_meta:
                    logger.warning(f"No metadata found for collection: {collection_name}")
                    collection_meta = {"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"}
            
            # Get embedding model from metadata
            embedding_model = collection_meta.get("embedding_model")
            if not embedding_model:
                embedding_model = config.get("training_config", {}).get(
                    "embeddingmodel", "sentence-transformers/all-MiniLM-L6-v2"
                )
            
            # 3. Generate embedding for search query
            query_embedding = self.embedding_manager.generate_embedding(
                [search_query], embedding_model
            )[0]

            # 4. Perform vector search with improved parameters
            search_start = time.time()
            vector_results = self.vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k * 2,  # Get more results for re-ranking
                filters=self._prepare_filters(filters, config.get("schema_mapping", {}), intent)
            )
            logger.info(f"Vector search completed in {time.time() - search_start:.3f}s")
            
            if not vector_results:
                logger.warning(f"No results found for query: {query}")
                
                # Check if collection is empty
                collection_stats = self.vector_store.get_collection_info(collection_name)
                vector_count = collection_stats.get("count", 0) if collection_stats else 0
                
                if vector_count == 0:
                    err_msg = f"No vectors found in collection {collection_name}. Please train the model with data first."
                    logger.error(err_msg)
                    return {
                        "error": err_msg,
                        "status": "empty_collection",
                        "search_metadata": {
                            "time_taken": time.time() - start_time,
                            "intent": intent,
                            "collection_stats": collection_stats
                        }
                    }
                
                # Return no results message with helpful suggestions
                suggestion = ""
                if intent["category"] == "food":
                    suggestion = "Try searching for specific kitchen appliances like 'air fryer' or 'non-stick pan'."
                elif intent["category"] == "mouse trap" or "mouse" in query:
                    suggestion = "Try searching for 'mouse trap' or 'rodent control'."
                
                return {
                    "results": [],
                    "generated_response": f"I couldn't find any products matching '{query}'. {suggestion}",
                    "search_metadata": {
                        "time_taken": time.time() - start_time,
                        "intent": intent,
                        "suggestion": suggestion
                    }
                }

            # 5. Format and improve results
            formatted_results = [
                self._format_search_result(result, config.get("schema_mapping", {}))
                for result in vector_results
            ]

            # 6. Apply hybrid scoring for better relevance
            texts = [
                f"{r['name']} {r['description']}" for r in formatted_results
            ]
            text_scores = self._compute_text_similarity(search_query, texts)
            
            # Combine scores: 70% vector, 30% text match
            for idx, result in enumerate(formatted_results):
                if idx < len(text_scores):
                    hybrid_score = 0.7 * result["score"] + 0.3 * text_scores[idx]
                    result["score"] = hybrid_score

            # 7. Re-rank results with cross-encoder if available
            if self.cross_encoder and len(formatted_results) > 1:
                rerank_start = time.time()
                reranked_results = self._rerank_results(search_query, formatted_results)
                formatted_results = reranked_results[:top_k]
                logger.info(f"Reranking completed in {time.time() - rerank_start:.3f}s")
            else:
                # Sort by score and limit results
                formatted_results = sorted(
                    formatted_results, key=lambda x: x["score"], reverse=True
                )[:top_k]

            # 8. Generate response with LLM
            llm_start = time.time()
            generated_response = self._generate_product_recommendations(
                query, formatted_results, intent
            )
            logger.info(f"LLM response generated in {time.time() - llm_start:.3f}s")

            # 9. Format final response
            response = {
                "generated_response": generated_response,
                "results": [self._format_result_for_frontend(r) for r in formatted_results],
                "search_metadata": {
                    "original_query": query,
                    "optimized_query": search_query,
                    "total_results": len(formatted_results),
                    "search_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                    "intent": intent
                }
            }

            # Cache the response
            self.response_cache[cache_key] = response

            logger.info(f"Total search time: {time.time() - start_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            time_taken = time.time() - start_time
            return {
                "error": str(e),
                "status": "failed",
                "time_taken": time_taken
            }

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using cross-encoder"""
        try:
            # Prepare pairs for cross-encoder
            pairs = []
            for result in results:
                # Combine name and description for better matching
                text = f"{result['name']} {result['description'][:200]}"
                pairs.append([query, text])
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Update scores and sort
            for idx, score in enumerate(scores):
                results[idx]["score"] = float(score)
                
            # Return sorted results
            return sorted(results, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            # Return original results if reranking fails
            return results

    def _format_search_result(self, result: Dict, schema_mapping: Dict) -> Dict:
        """Format search result with better structure"""
        try:
            metadata = result.get("metadata", {})
            custom_metadata = metadata.get("custom_metadata", {})
            
            formatted = {
                "mongo_id": metadata.get("mongo_id", ""),
                "score": round(float(result.get("score", 0.0)), 4),
                "name": metadata.get("name", ""),
                "description": metadata.get("description", ""),
                "category": metadata.get("category", ""),
                "metadata": custom_metadata,
                "qdrant_id": result.get("id", ""),
            }
            
            # Add price if available
            if "price" in custom_metadata:
                formatted["price"] = custom_metadata["price"]
            elif "discount_price" in custom_metadata:
                formatted["price"] = custom_metadata["discount_price"]
            
            return formatted
        except Exception as e:
            logger.error(f"Result formatting error: {e}")
            return {"error": "Result formatting failed"}

    def _format_result_for_frontend(self, result: Dict) -> Dict:
        """Format result for frontend display"""
        try:
            metadata = result.get("metadata", {})
            return {
                "id": result.get("mongo_id", ""),
                "name": result.get("name", ""),
                "description": result.get("description", ""),
                "category": result.get("category", ""),
                "score": round(float(result.get("score", 0.0)), 4),
                **{k: str(v) for k, v in metadata.items()},
                "url": f"/product/{result.get('mongo_id', '')}"
            }
        except Exception as e:
            logger.error(f"Frontend formatting error: {e}")
            return {"error": "Formatting failed"}

    def _generate_product_recommendations(self, query: str, results: List[Dict], intent: Dict) -> str:
        """Generate tailored product recommendations with enhanced prompting"""
        if not results:
            return f"I couldn't find any products matching '{query}'."
            
        try:
            # Create product context with the most relevant information
            product_context = []
            for i, product in enumerate(results[:3]):
                price = product.get("price", "N/A")
                if isinstance(price, str) and not price.startswith("$"):
                    price = f"${price}"
                    
                product_context.append(
                    f"{i+1}. {product['name']} - {product['description'][:150]}... "
                    f"Price: {price}, Ratings: {product.get('metadata', {}).get('ratings', 'N/A')}"
                )
                
            product_context_str = "\n".join(product_context)
            
            # Create prompt for LLM
            prompt = f"""<|user|>
            I'm looking for: {query}
            
            Available relevant products:
            {product_context_str}

            Please recommend products considering:
            1. Price vs value analysis
            2. Feature match to query
            3. Popularity signals
            4. Concise natural language response

            <|assistant|>
            """
            
            # Generate response with corrected parameters (fix warnings)
            response = self.llm_manager.generate_response(prompt)
            return response
                    
        except Exception as e:
            logger.error(f"Product recommendation generation failed: {e}")
            return f"I couldn't generate product recommendations. Error: {str(e)}"

    def _prepare_filters(self, filters: Dict, schema_mapping: Dict, intent: Dict) -> Dict:
        """Convert frontend filters to vector store format with intent consideration"""
        if not filters:
            filters = {}
        
        prepared_filters = {}
        filter_fields = schema_mapping.get("filter_fields", [])
        
        for field, value in filters.items():
            if field in filter_fields:
                if isinstance(value, list):
                    prepared_filters[f"metadata.{field}"] = {"$in": value}
                else:
                    prepared_filters[f"metadata.{field}"] = value
        
        # Add intent-based filters
        if intent["category"] and intent["category"] != "general":
            prepared_filters["metadata.category"] = intent["category"]
            
        return prepared_filters


search_service = OptimizedSearchService()

@app.route("/search", methods=["POST"])
def search():
    """Search endpoint with API config support"""
    try:
        data = request.get_json()
        if not data or "query" not in data or "model_path" not in data:
            return jsonify({"error": "Missing required fields"}), 400

        logger.info(f"Received search request for query: {data['query']}")
        with ThreadPoolExecutor() as executor:
            future = executor.submit(
                search_service.search,
                query=data["query"],
                model_path=data["model_path"],
                top_k=data.get("max_items", 20),
                filters=data.get("filters", {}),
            )
            response = future.result(timeout=500)

        logger.info("Search request completed successfully")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}", exc_info=True)
        return jsonify({"error": "Search failed", "message": str(e)}), 500        

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "models_loaded": list(
                search_service.embedding_manager.embedding_models.keys()
            ),
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=AppConfig.SERVICE_PORT, debug=False)


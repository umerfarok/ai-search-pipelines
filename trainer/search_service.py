from concurrent.futures import ThreadPoolExecutor
import os
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import requests
from config import AppConfig
from vector_store import VectorStore  # Updated import
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
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Any
from sentence_transformers import CrossEncoder
import torch
from vector_store import VectorStore
from simple_cache import TimedCache  # Change SimpleCache to TimedCache
from training_service import EmbeddingManager
from config import AppConfig
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class EmbeddingManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_models = {}
        self.model_load_lock = threading.Lock()
        self.max_seq_length = 512
        self.cache_dir = AppConfig.TRANSFORMER_CACHE

    def get_model(self, model_path: str):
        if (model := self.embedding_models.get(model_path)) is not None:
            return model

        try:
            with self.model_load_lock:
                if model_path not in self.embedding_models:
                    model = SentenceTransformer(
                        model_path, cache_folder=self.cache_dir, device=self.device
                    )
                    self.embedding_models[model_path] = model
                return self.embedding_models[model_path]
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise

    def generate_embedding(self, text: List[str], model_path: str) -> np.ndarray:
        if not text:
            raise ValueError("Empty text input")

        model = self.get_model(model_path)
        batch_size = 128

        try:
            with torch.no_grad():
                embeddings = model.encode(
                    text,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
                return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class UnifiedLLMManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"
        self.model = None
        self.tokenizer = None
        self.model_lock = threading.Lock()
        self._initialize_model()

    def _initialize_model(self):
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            logger.warning("Hugging Face token not found!")

        try:
            logger.info(f"Loading model: {self.model_name}")
            quant_config = (
                BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    trust_remote_code=True,
                )
                if torch.cuda.is_available()
                else None
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto",
                quantization_config=quant_config,
                token=hf_token,
                trust_remote_code=True,
            )
            logger.info(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise

    @contextlib.contextmanager
    def get_model(self):
        with self.model_lock:
            yield self.model, self.tokenizer

    def generate_streamed(self, prompt: str, max_length: int = 1024) -> Queue:
        output_queue = Queue()
        streamer = TextIteratorStreamer(self.tokenizer)

        def generate():
            try:
                with self.get_model() as (model, tokenizer):
                    inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                    model.generate(
                        **inputs,
                        max_length=max_length,
                        streamer=streamer,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.95,
                    )
            except Exception as e:
                output_queue.put({"error": str(e)})

        Thread(target=generate).start()
        for text in streamer:
            output_queue.put({"text": text})
        output_queue.put(None)
        return output_queue


class EnhancedSearchService:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_manager = None
        try:
            self.llm_manager = UnifiedLLMManager()
            logger.info("LLM Manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Manager: {e}")
        self.response_cache = TTLCache(maxsize=1000, ttl=3600)
        self.query_cache = TTLCache(maxsize=1000, ttl=3600)
        self.embedding_cache = TTLCache(maxsize=10000, ttl=3600)  # Cache embeddings for 1 hour
        self.semantic_cache = TTLCache(maxsize=1000, ttl=1800)    # Cache similar queries for 30 mins
        self.config_cache = TTLCache(maxsize=100, ttl=300)        # Cache configs for 5 mins
        self.query_similarity_threshold = 0.85

    def _cache_key(self, query: str, model_path: str) -> str:
        """Generate cache key for query+model combination"""
        return f"{model_path}:{query}"

    def _get_similar_cached_query(self, query: str, model_path: str) -> Optional[Dict]:
        """Find semantically similar cached query results"""
        if not self.semantic_cache:
            return None
            
        query_embedding = self._get_cached_embedding(query, model_path)
        if query_embedding is None:
            return None
            
        for cached_query, cached_results in self.semantic_cache.items():
            cached_embedding = self._get_cached_embedding(cached_query, model_path)
            if cached_embedding is not None:
                similarity = np.dot(query_embedding, cached_embedding)
                if similarity > self.query_similarity_threshold:
                    return cached_results
        return None

    def _get_cached_embedding(self, text: str, model_path: str) -> Optional[np.ndarray]:
        """Get cached embedding or generate new one"""
        cache_key = self._cache_key(text, model_path)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        embedding = self.embedding_manager.generate_embedding([text], model_path)
        if embedding is not None:
            self.embedding_cache[cache_key] = embedding[0]
            return embedding[0]
        return None

    def _fetch_config(self, model_path: str) -> Dict:
        """Fetch configuration from API with fallback to local cache."""
        try:
            if model_path in self.config_cache:
                return self.config_cache[model_path]
                
            # Extract config_id from model_path
            config_id = model_path.split("/")[-1]
            
            # Try API first
            url = f"{AppConfig.API_HOST}/config/{config_id}"
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                config = response.json()
                logger.info(f"Fetched config for ID: {config_id} from API")
                self.config_cache[model_path] = config
                
                # Save to local cache file for future use
                self._save_config_to_cache(config_id, config)
                
                return config
            except requests.RequestException as e:
                logger.error(f"Failed to fetch config from {url}: {e}")
                
                # Try to load from local cache
                cached_config = self._load_config_from_cache(config_id)
                if cached_config:
                    logger.info(f"Using cached config for ID: {config_id}")
                    self.config_cache[model_path] = cached_config
                    return cached_config
                
                # Infer config from vector store metadata
                collection_name = f"products_{config_id}"
                collection_meta = self.vector_store.get_collection_metadata(collection_name)
                
                if collection_meta:
                    logger.info(f"Using metadata-derived config for ID: {config_id}")
                    inferred_config = {
                        "id": config_id,
                        "training_config": {
                            "embeddingmodel": collection_meta.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
                        },
                        "schema_mapping": collection_meta.get("schema_mapping", {})
                    }
                    self.config_cache[model_path] = inferred_config
                    return inferred_config
                
                # If all else fails, use default config
                logger.warning(f"Using default config for ID: {config_id}")
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
            logger.error(f"Error in _fetch_config: {e}")
            # Return minimal default config as last resort
            return {
                "id": model_path.split("/")[-1],
                "training_config": {
                    "embeddingmodel": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "schema_mapping": {
                    "customcolumns": []
                }
            }

    def _save_config_to_cache(self, config_id: str, config: Dict) -> None:
        """Save config to local cache file."""
        try:
            cache_dir = os.path.join("/app", "cache", "configs")
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, f"{config_id}.json")
            with open(cache_file, 'w') as f:
                json.dump(config, f)
                
            logger.info(f"Saved config to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save config to cache: {e}")

    def _load_config_from_cache(self, config_id: str) -> Optional[Dict]:
        """Load config from local cache file."""
        try:
            cache_file = os.path.join("/app", "cache", "configs", f"{config_id}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    config = json.load(f)
                return config
            return None
        except Exception as e:
            logger.warning(f"Failed to load config from cache: {e}")
            return None

    def _expand_query(self, query: str) -> str:
        """Expand query using LLM"""
        cache_key = f"query_expansion:{query}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        prompt = f"""As a search query expansion expert, enhance this product search query:
        Query: {query}
        
        Instructions:
        1. Identify key concepts and intent
        2. Add relevant synonyms and related terms
        3. Include product attributes and categories
        4. Keep expansion focused and relevant
        
        Enhanced query:"""

        try:
            if not self.llm_manager:
                return query
            response_queue = self.llm_manager.generate_streamed(prompt)
            expanded_query = ""
            while True:
                item = response_queue.get()
                if item is None:
                    break
                if "error" in item:
                    raise Exception(item["error"])
                expanded_query += item["text"]
            final_query = f"{query} {expanded_query.strip()}"
            self.query_cache[cache_key] = final_query
            return final_query
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return query

    def _semantic_search(
        self, query: str, model_path: str, top_k: int = 5
    ) -> List[Dict]:
        """Perform semantic search using vector database with better error handling."""
        try:
            # Get config - new robust version will always return something
            config = self._fetch_config(model_path)
            
            embedding_model = config.get("training_config", {}).get(
                "embeddingmodel", 
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            config_id = model_path.split("/")[-1]
            collection_name = f"products_{config_id}"

            # Verify collection exists
            if not self.vector_store.collection_exists(collection_name):
                logger.error(f"Collection not found: {collection_name}")
                return []

            # Generate query embedding
            try:
                query_embedding = self.embedding_manager.generate_embedding([query], embedding_model)
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                # Try fallback model
                try:
                    fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                    logger.info(f"Trying fallback embedding model: {fallback_model}")
                    query_embedding = self.embedding_manager.generate_embedding([query], fallback_model)
                except Exception as e2:
                    logger.error(f"Fallback embedding also failed: {e2}")
                    return []

            # Perform vector search
            results = self.vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding[0].tolist(),
                limit=top_k,
            )
            
            # Format results
            schema_mapping = config.get("schema_mapping", {"customcolumns": []})
            formatted_results = [
                self._format_search_result(result, schema_mapping)
                for result in results
            ]
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _format_search_result(self, result: Dict, schema_mapping: Dict) -> Dict:
        """Format search result using schema mapping from Qdrant payload"""
        try:
            metadata = result.get("metadata", {})
            custom_metadata = metadata.get("custom_metadata", {})
            expected_cols = [
                col["name"] for col in schema_mapping.get("customcolumns", [])
            ]
            missing_cols = [col for col in expected_cols if col not in custom_metadata]
            if missing_cols:
                logger.warning(
                    f"Missing metadata columns in result {result.get('id')}: {missing_cols}"
                )

            return {
                "mongo_id": metadata.get("mongo_id", ""),
                "score": round(float(result.get("score", 0.0)), 4),
                "name": metadata.get("name", ""),
                "description": metadata.get("description", ""),
                "category": metadata.get("category", ""),
                "metadata": custom_metadata,
                "qdrant_id": result.get("id", ""),
            }
        except Exception as e:
            logger.error(f"Result formatting error: {str(e)}")
            return {"error": "Result formatting failed"}

    def _format_result_for_frontend(self, result: Dict) -> Dict:
        """Format a single result for frontend display"""
        try:
            metadata = result.get("metadata", {})
            return {
                "id": result.get("mongo_id", ""),
                "name": result.get("name", ""),
                "description": result.get("description", ""),
                "category": result.get("category", ""),
                "score": round(float(result.get("score", 0.0)), 4),
                **{
                    k: str(v) for k, v in metadata.items()
                },
                "url": f"/product/{result.get('mongo_id', '')}",
            }
        except Exception as e:
            logger.error(f"Result formatting error: {e}")
            return {"error": "Formatting failed"}

    def _generate_response(self, query: str, results: List[Dict]) -> str:
        """Generate response using LLM"""
        if not self.llm_manager:
            return self._generate_fallback_response(results)

        if not isinstance(results, list):
            logger.error(f"Expected 'results' to be a list, got {type(results)}")
            return self._generate_fallback_response(results)

        product_context = "\n".join(
            [
                f"- {p['name']} ({p.get('category', 'N/A')}): "
                f"{p.get('description', '')[:150]}... "
                f"Price: {p.get('metadata', {}).get('discount_price', 'N/A')} "
                f"Ratings: {p.get('metadata', {}).get('ratings', 'N/A')} "
                for p in results[:3]
            ]
        )

        prompt = f"""<|user|>
        I'm looking for: {query}

        Available relevant products:
        {product_context}

        Please recommend products considering:
        1. Price vs value analysis
        2. Feature match to query
        3. Popularity signals
        4. Concise natural language response

        <|assistant|>
        """

        try:
            logger.info("Starting response generation")
            with self.llm_manager.get_model() as (model, tokenizer):
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info("Response generation completed")
                return response.split("<|assistant|>")[-1].strip()
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return self._generate_fallback_response(results)

    def _generate_fallback_response(self, results: List[Dict]) -> str:
        """Generate a simple structured response when LLM fails"""
        try:
            if not isinstance(results, list):
                raise ValueError(f"Expected results to be a list, got {type(results)}")
            response = "Here are some relevant products:\n\n"
            for p in results[:3]:
                metadata = p.get("metadata", {})
                price = metadata.get("discount_price", "Price not available")
                ratings = metadata.get("ratings", "No ratings")
                response += f"- {p.get('name', 'Unknown')}\n"
                response += f"  Price: {price}\n"
                response += f"  Ratings: {ratings}\n\n"
            return response
        except Exception as e:
            logger.error(f"Fallback response generation failed: {e}")
            return "Unable to generate product recommendations."

    def _format_frontend_response(
        self, results: List[Dict], generated_text: str, query: str, expanded_query: str
    ) -> Dict:
        """Structure response for frontend consumption"""
        try:
            return {
                "generated_response": generated_text,
                "results": [self._format_result_for_frontend(res) for res in results],
                "search_metadata": {
                    "original_query": query,
                    "expanded_query": expanded_query,
                    "total_results": len(results),
                    "timestamp": datetime.now().isoformat(),
                },
            }
        except Exception as e:
            logger.error(f"Failed to format frontend response: {e}")
            return {"error": "Failed to format response", "message": str(e)}

    def _preprocess_query(self, query: str) -> str:
        """Clean and normalize search query"""
        query = query.lower().strip()
        query = re.sub(r"[^\w\s-]", "", query)
        return query

    def _apply_filters(self, results: List[Dict], filters: Dict) -> List[Dict]:
        """Apply post-search filters"""
        filtered = []
        for res in results:
            metadata = res.get("metadata", {})
            match = True
            if "max_price" in filters:
                price_str = metadata.get("discount_price", "0")
                price = float(re.sub(r"[^\d.]", "", price_str)) if price_str else 0
                if price > filters["max_price"]:
                    match = False
            if "categories" in filters:
                if res.get("category", "").lower() not in [
                    c.lower() for c in filters["categories"]
                ]:
                    match = False
            if match:
                filtered.append(res)
        return filtered

    def search(
        self, query: str, model_path: str, top_k: int = 5, filters: Dict = None
    ) -> Dict:
        """Enhanced search with better configuration integration"""
        try:
            # Preprocess query
            query = self._preprocess_query(query)
            
            # Get config_id from model_path
            config_id = model_path.split("/")[-1]
            collection_name = f"products_{config_id}"
            
            # Verify collection exists
            if not self.vector_store.collection_exists(collection_name):
                logger.error(f"Collection {collection_name} not found")
                return {
                    "error": "Model not found",
                    "status": "failed"
                }

            # Get collection metadata to determine correct embedding model
            collection_meta = self.vector_store.get_collection_metadata(collection_name)
            if not collection_meta:
                raise ValueError("Collection metadata not found")

            # Use the embedding model specified in collection metadata
            embedding_model = collection_meta.get("embedding_model")
            if not embedding_model:
                embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

            # Check cache for similar queries
            cache_key = self._cache_key(query, embedding_model)
            if cache_key in self.response_cache:
                logger.info("Returning cached results")
                return self.response_cache[cache_key]

            # Expand query with domain awareness
            expanded_query = self._expand_query(query)
            logger.info(f"Expanded query: '{query}' -> '{expanded_query}'")

            # Generate embedding using correct model
            query_embedding = self._get_cached_embedding(expanded_query, embedding_model)
            if query_embedding is None:
                query_embedding = self.embedding_manager.generate_embedding(
                    [expanded_query], 
                    embedding_model
                )[0]
                self.embedding_cache[cache_key] = query_embedding

            # Perform vector search with proper parameter naming
            vector_results = self.vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                limit=top_k,  # Use limit instead of top_k
                filters=self._prepare_filters(filters, collection_meta.get("schema_mapping", {}))
            )

            # Format results according to schema mapping
            formatted_results = [
                self._format_search_result(result, collection_meta["schema_mapping"])
                for result in vector_results
            ]

            # Generate response with LLM
            generated_response = self._generate_response(query, formatted_results)

            # Prepare final response
            response = self._format_frontend_response(
                formatted_results,
                generated_response,
                query,
                expanded_query
            )

            # Cache the response
            self.response_cache[cache_key] = response
            
            return response

        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "status": "failed"
            }

    def _prepare_filters(self, filters: Dict, schema_mapping: Dict) -> Dict:
        """Convert frontend filters to vector store format"""
        if not filters:
            return {}

        prepared_filters = {}
        filter_fields = schema_mapping.get("filter_fields", [])
        
        for field, value in filters.items():
            if field in filter_fields:
                if isinstance(value, list):
                    prepared_filters[f"metadata.{field}"] = {"$in": value}
                else:
                    prepared_filters[f"metadata.{field}"] = value

        return prepared_filters


class SearchService:
    def __init__(self):
        self.vector_store = VectorStore()
        self.embedding_manager = EmbeddingManager()
        self.cache = TimedCache(ttl_seconds=300)  # Use TimedCache instead
        self.cross_encoder = None
        self.use_cross_encoder = True
        self.hybrid_search = True
        
        # Initialize NLP if available
        try:
            self.nlp = spacy.load("en_core_web_md")
        except:
            logger.warning("Could not load spaCy model")
            self.nlp = None
        
        self.domain_keywords = AppConfig.DOMAIN_KEYWORDS
        self._load_cross_encoder()

    def search(self, config_id: str, query: str, top_k: int = 10, 
               threshold: float = 0.5, use_hybrid: bool = True,
               rerank: bool = True) -> List[Dict]:
        """
        Perform semantic search with enhanced relevance using updated config structure
        """
        start_time = time.time()
        
        if not query or not config_id:
            logger.error("Missing required search parameters")
            return []
        
        collection_name = f"products_{config_id}"
        
        # Get collection metadata
        collection_meta = self.vector_store.get_collection_metadata(collection_name)
        if not collection_meta:
            logger.error(f"Collection {collection_name} not found")
            return []
            
        # Use embedding model from collection metadata
        model_name = collection_meta.get("embedding_model")
        if not model_name:
            logger.error("No embedding model specified in collection metadata")
            return []

        # Process and expand query
        expanded_query = self.expand_query(query)
        logger.info(f"Expanded query: {expanded_query}")
        
        # Generate embedding using the correct model
        query_embedding = self.embedding_manager.generate_embedding(
            [expanded_query], 
            model_name
        )
        
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []

        # Perform vector search with proper metadata structure
        results = self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_embedding[0],
            top_k=top_k,
            threshold=threshold
        )

        if not results:
            return []

        # Format results according to updated schema
        formatted_results = []
        for item in results:
            try:
                metadata = item.get("metadata", {})
                result = {
                    "id": metadata.get("id"),
                    "name": metadata.get("name"),
                    "description": metadata.get("description"),
                    "category": metadata.get("category"),
                    "score": float(item.get("score", 0.0)),
                    "metadata": metadata.get("custom_metadata", {})
                }
                formatted_results.append(result)
            except Exception as e:
                logger.error(f"Error formatting result: {e}")
                continue

        # Apply cross-encoder reranking if enabled
        if rerank and self.cross_encoder and len(formatted_results) > 1:
            try:
                reranked_results = self._rerank_results(
                    query, 
                    formatted_results
                )
                formatted_results = reranked_results
            except Exception as e:
                logger.error(f"Reranking failed: {e}")

        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.3f}s")
        
        return formatted_results

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """Rerank results using cross-encoder with updated metadata structure"""
        pairs = []
        for result in results:
            text = f"{result['name']} {result['description']}"
            pairs.append([query, text])

        scores = self.cross_encoder.predict(pairs)
        
        # Update scores and sort
        for idx, score in enumerate(scores):
            results[idx]["score"] = float(score)
            
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def expand_query(self, query: str) -> str:
        """Enhanced query expansion with domain awareness"""
        if not query:
            return query

        expanded_terms = []
        
        # Add domain-specific terms
        for domain, terms in self.domain_keywords.items():
            if any(kw in query.lower() for kw in terms):
                expanded_terms.extend(terms[:3])  # Add top 3 related terms
        
        # Use NLP for semantic expansion if available
        if self.nlp:
            doc = self.nlp(query)
            for token in doc:
                if token.pos_ in ["NOUN", "ADJ"] and token.has_vector:
                    # Find similar words using word vectors
                    similars = token.similarity(doc)
                    if similars > 0.7:
                        expanded_terms.append(token.text)

        # Remove duplicates and combine
        expanded_terms = list(set(expanded_terms))
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
            logger.info(f"Expanded query: {query} -> {expanded_query}")
            return expanded_query

        return query

    def get_collection_stats(self, config_id: str) -> Dict:
        """Get statistics for a collection"""
        collection_name = f"products_{config_id}"
        
        try:
            # Get collection metadata
            metadata = self.vector_store.get_collection_metadata(collection_name)
            if not metadata:
                return {"error": "Collection not found"}
                
            # Get additional statistics
            stats = self.vector_store.get_collection_stats(collection_name)
            
            return {
                "config_id": config_id,
                "metadata": metadata,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}


def load_model(model_path: str) -> Any:
    """Load a trained model from the specified path."""
    # Ensure model_path is correctly passed from ModelConfig.ModelPath
    # ...existing code...


def search(model, query: str, top_k: int = 5) -> list:
    """Perform a search using the specified model."""
    # ...existing code...


search_service = EnhancedSearchService()


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

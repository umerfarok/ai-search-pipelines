import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from flask import Flask, request, jsonify
import requests
import spacy
from threading import Thread
from queue import Queue
import contextlib
import torch.cuda
from cachetools import TTLCache
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import time
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.corpus import wordnet
from collections import Counter

from vector_store import VectorStore
from config import AppConfig

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Change logging level for better focus on important messages
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)

import re
from domain_knowledge import (
    PRODUCT_CATEGORIES, 
    PRODUCT_FEATURES,
    DOMAIN_TERM_EXPANSION,
    QUERY_INTENTS,
    get_expanded_terms,
    get_related_categories,
    get_feature_terms
)

class DynamicSearchProcessor:
    """Professional-grade dynamic search processor that adapts to any product domain"""
    
    def __init__(self, embedding_manager=None):
        """Initialize with flexible components"""
        self.embedding_manager = embedding_manager
        # Initialize caches
        self.intent_cache = TTLCache(maxsize=1000, ttl=3600)
        self.expansion_cache = TTLCache(maxsize=1000, ttl=3600)
        
        # Dynamic category discovery 
        self.product_embeddings = {}
        self.discovered_categories = set()
        
        # Load resources if needed
        self._load_resources()
        
        # Zero-shot classifier for flexible categorization
        try:
            self.zero_shot_model = None
            self.zero_shot_tokenizer = None
            # We'll initialize this on demand to save resources
        except Exception as e:
            logger.warning(f"Could not initialize zero-shot classifier: {e}")
            
        # Load nltk resources 
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
        try:
            self.stopwords = set(nltk.corpus.stopwords.words('english'))
        except:
            self.stopwords = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
                              "you", "your", "yours", "yourself", "yourselves", "he", "him", 
                              "his", "himself", "she", "her", "hers", "herself", "it", "its", 
                              "itself", "they", "them", "their", "theirs", "themselves", "what", 
                              "which", "who", "whom", "this", "that", "these", "those", "am", 
                              "is", "are", "was", "were", "be", "been", "being", "have", "has", 
                              "had", "having", "do", "does", "did", "doing", "a", "an", "the", 
                              "and", "but", "if", "or", "because", "as", "until", "while", "of", 
                              "at", "by", "for", "with", "about", "against", "between", "into", 
                              "through", "during", "before", "after", "above", "below", "to", 
                              "from", "up", "down", "in", "out", "on", "off", "over", "under", 
                              "again", "further", "then", "once", "here", "there", "when", 
                              "where", "why", "how", "all", "any", "both", "each", "few", 
                              "more", "most", "other", "some", "such", "no", "nor", "not", 
                              "only", "own", "same", "so", "than", "too", "very", "s", "t", 
                              "can", "will", "just", "don", "should", "now"}
    
    def _load_resources(self):
        """Load necessary resources for query understanding"""
        # Load minimal spaCy model if available
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                # Fall back to simpler model
                self.nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
            logger.info("Loaded spaCy model for language understanding")
        except:
            self.nlp = None
            logger.warning("SpaCy not available for enhanced language understanding")
    
    def _get_zero_shot_classifier(self):
        """Load zero-shot classifier on demand to save memory"""
        if self.zero_shot_model is None:
            try:
                # Use a small, efficient model for zero-shot classification
                model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
                
                self.zero_shot_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.zero_shot_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.zero_shot_model.to("cuda")
                    # Use half precision for efficiency
                    self.zero_shot_model.half()
                
                logger.info("Zero-shot classifier loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load zero-shot classifier: {e}")
                # Return None to indicate failure
                return None
                
        return self.zero_shot_model, self.zero_shot_tokenizer
    
    def detect_intent(self, query: str, products: List[Dict] = None) -> Dict[str, Any]:
        """Dynamically detect search intent using domain knowledge and semantic understanding"""
        # Check cache first
        if query in self.intent_cache:
            return self.intent_cache[query]
        
        # Basic intent features
        intent_data = {
            "category": None,
            "confidence": 0.0,
            "is_question": '?' in query or query.lower().startswith(('how', 'what', 'where', 'do you', 'can i', 'is there')),
            "keywords": [],
            "action": "find",  # Default action
            "domain_terms": [],
            "product_features": []
        }
        
        # Extract keywords (excluding stopwords)
        words = query.lower().split()
        intent_data["keywords"] = [w for w in words if len(w) > 2 and w not in self.stopwords]
        
        # Detect action type using domain knowledge
        for intent_type, signals in QUERY_INTENTS.items():
            if any(signal in query.lower() for signal in signals):
                intent_data["action"] = intent_type
                break
        
        # Enhanced domain-specific category detection
        detected_category = self._detect_domain_category(query)
        if detected_category:
            intent_data["category"] = detected_category
            intent_data["confidence"] = 0.8
            intent_data["source"] = "domain_knowledge"
            
            # Add domain-specific feature terms
            intent_data["product_features"] = get_feature_terms(detected_category)
            
        # If we have products but no category, try to detect from products
        elif products:
            category = self._detect_category_from_products(query, products)
            if category:
                intent_data["category"] = category
                intent_data["confidence"] = 0.7
                intent_data["source"] = "product_analysis"
        
        # Use context clues from the query to find domain-specific terms
        domain_terms = []
        for term in intent_data["keywords"]:
            # Check for domain-specific terms
            expanded_terms = get_expanded_terms(term)
            if expanded_terms:
                domain_terms.extend(expanded_terms)
                
            # Check for related categories
            related_categories = get_related_categories(term)
            if related_categories:
                for category in related_categories:
                    if not intent_data["category"]:  # Only set category if none detected yet
                        intent_data["category"] = category
                        intent_data["confidence"] = 0.6
                        intent_data["source"] = "related_category"
                    # Add feature terms for this category
                    intent_data["product_features"].extend(get_feature_terms(category))
        
        # Add domain terms to intent data
        intent_data["domain_terms"] = list(set(domain_terms))
        
        # Special case handling for water filtering/purification
        if any(water_term in query.lower() for water_term in ["water", "drink", "drinking", "clean"]) and \
           any(filter_term in query.lower() for filter_term in ["filter", "purify", "clean", "pure", "safe"]):
            intent_data["category"] = "water filter"
            intent_data["confidence"] = 0.9
            intent_data["source"] = "combined_terms"
            intent_data["product_features"] = get_feature_terms("water filter")
            
        # Special case for outdoor scenarios
        if any(outdoor_term in query.lower() for outdoor_term in ["jungle", "wilderness", "forest", "camping", "hiking", "survival", "emergency"]):
            # Add outdoor context
            if "outdoor_context" not in intent_data:
                intent_data["outdoor_context"] = True
                
            # If water related, boost water filter intent
            if any(water_term in query.lower() for water_term in ["water", "drink", "drinking"]):
                intent_data["category"] = "water filter"
                intent_data["confidence"] = 0.95
                intent_data["source"] = "outdoor_water_need"
                intent_data["product_features"] = get_feature_terms("water filter")
                intent_data["domain_terms"].extend(["portable", "survival", "emergency", "wilderness"])
        
        # Add the detected intent to cache
        self.intent_cache[query] = intent_data
        return intent_data

    def _detect_domain_category(self, query: str) -> str:
        """Detect product category using domain knowledge"""
        query_lower = query.lower()
        
        # Direct category matching
        for category, terms in PRODUCT_CATEGORIES.items():
            # Check if category is directly mentioned
            if category.lower() in query_lower:
                return category
                
            # Check if any associated terms are in the query
            term_matches = [term for term in terms if term.lower() in query_lower]
            if term_matches:
                return category
                
        # Feature-based category matching
        for category, features in PRODUCT_FEATURES.items():
            feature_matches = []
            
            # Check all feature types
            for feature_type, terms in features.items():
                for term in terms:
                    if term.lower() in query_lower:
                        feature_matches.append(term)
                        
            if feature_matches:
                return category
                
        return None

    def _detect_category_from_products(self, query: str, products: List[Dict]) -> str:
        """Detect category by comparing query with available products"""
        if not products:
            return None
            
        # Collect categories from products
        categories = {}
        for product in products:
            cat = product.get("category", "").lower()
            if cat:
                categories[cat] = categories.get(cat, 0) + 1
        
        if not categories:
            return None
            
        # Return the most common category as a default approach
        most_common = max(categories.items(), key=lambda x: x[1])
        return most_common[0]

    def _generate_potential_categories(self, keywords: List[str]) -> List[str]:
        """Generate potential categories based on query keywords"""
        # Start with general product categories
        general_categories = [
            "electronics", "home appliances", "furniture", "kitchen", 
            "cleaning", "pet supplies", "clothing", "toys", "tools", 
            "office supplies", "health", "beauty", "food", "sports", 
            "water filter", "pest control"
        ]
        
        # Add categories from discovered categories
        all_categories = list(general_categories) + list(self.discovered_categories)
        
        # Generate compound categories using keywords
        keyword_categories = []
        for keyword in keywords:
            if len(keyword) > 3:  # Only use meaningful keywords
                for base in ["products", "items", "appliances", "tools", "supplies"]:
                    keyword_categories.append(f"{keyword} {base}")
                    
        # Combine all potential categories
        potential_categories = all_categories + keyword_categories
        
        # Deduplicate
        return list(set(potential_categories))

    def _zero_shot_classify(self, query: str, candidate_labels: List[str]) -> Tuple[str, float]:
        """Classify query using zero-shot classification"""
        try:
            model, tokenizer = self._get_zero_shot_classifier()
            if not model or not tokenizer:
                return None, 0.0
                
            # Limit number of categories to avoid excessive computation
            if len(candidate_labels) > 10:
                candidate_labels = candidate_labels[:10]

            # Format for zero-shot NLI task
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sequences = [f"{query}. {label}." for label in candidate_labels]
            
            # Tokenize
            inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt").to(device)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                scores = scores[:, 0].tolist()  # Get entailment scores
            
            # Find best match
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            best_category = candidate_labels[best_idx]
            
            return best_category, best_score
        except Exception as e:
            logger.error(f"Zero-shot classification error: {e}")
            return None, 0.0

    def expand_query(self, query: str, intent: Dict = None) -> str:
        """Expand query semantically without relying on hardcoded rules"""
        # Check cache first
        if query in self.expansion_cache:
            return self.expansion_cache[query]
            
        # Tokenize and normalize
        tokens = nltk.word_tokenize(query.lower()) if hasattr(nltk, 'word_tokenize') else query.lower().split()
        
        # Extract key terms (excluding stopwords)
        key_terms = [term for term in tokens if term not in self.stopwords and len(term) > 2]
        if not key_terms:
            return query  # No meaningful terms to expand
            
        # Expansion techniques:
        # 1. Synonym expansion via WordNet
        synonyms = []
        for term in key_terms:
            term_synonyms = self._get_wordnet_synonyms(term)
            # Take up to 2 synonyms per term to avoid dilution
            synonyms.extend(term_synonyms[:2])
            
        # 2. Handle multi-word concepts
        phrases = self._extract_phrases(query)
        for phrase in phrases:
            # Try to find phrase synonyms
            phrase_synonyms = self._get_wordnet_synonyms(phrase)
            synonyms.extend(phrase_synonyms[:2])
            
        # 3. Add intent-based expansion
        if intent and intent.get("category") and intent.get("confidence", 0) > 0.6:
            category_terms = intent["category"].split()
            for term in category_terms:
                if term not in tokens and term not in self.stopwords and len(term) > 2:
                    synonyms.append(term)
        
        # Deduplicate and filter expansions
        expanded_terms = []
        for term in synonyms:
            # Only add if not in original query and not a stopword
            if term not in query.lower() and term not in self.stopwords:
                expanded_terms.append(term)
                
        # Limit expansion size
        if len(expanded_terms) > 5:
            expanded_terms = expanded_terms[:5]
            
        # Create expanded query
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
            logger.info(f"Expanded query: '{query}' -> '{expanded_query}'")
        else:
            expanded_query = query
            
        # Cache the result
        self.expansion_cache[query] = expanded_query
        return expanded_query
        
    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text"""
        # Use spaCy if available for better phrase extraction
        if self.nlp:
            doc = self.nlp(text)
            phrases = []
            
            # Extract noun phrases
            for np in doc.noun_chunks:
                if len(np.text) > 3:
                    phrases.append(np.text.lower())
                    
            # Extract named entities
            for ent in doc.ents:
                if len(ent.text) > 3:
                    phrases.append(ent.text.lower())
                    
            return phrases
        else:
            # Fallback to simple n-gram approach
            words = text.lower().split()
            phrases = []
            
            # Add bigrams
            if len(words) >= 2:
                for i in range(len(words)-1):
                    phrases.append(f"{words[i]} {words[i+1]}")
                    
            return phrases
        
    def _get_wordnet_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word/phrase using WordNet"""
        synonyms = []
        
        # Handle multi-word phrases differently
        if " " in word:
            # For phrases, try to get synonyms for each component word
            parts = word.split()
            for part in parts:
                if len(part) > 3 and part not in self.stopwords:  # Only process meaningful words
                    part_synonyms = self._get_wordnet_synonyms(part)
                    synonyms.extend(part_synonyms)
            return synonyms[:3]  # Limit to top 3 synonyms for phrases

        # Single word processing
        for syn in wordnet.synsets(word):
            # Only use the first sense (most common)
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != word and synonym not in synonyms:
                    synonyms.append(synonym)
                    
            # Only process the first synset for efficiency
            break
            
        return synonyms[:3]  # Limit to 3 synonyms per word

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


# Import the enhanced LLM manager
from simple_llm import EnhancedLLMManager

class OptimizedSearchService:
    """Improved search service with better performance and relevance"""
    
    def __init__(self):
        # Initialize optimized components
        self.embedding_manager = OptimizedEmbeddingManager()
        self.vector_store = VectorStore()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to initialize the enhanced LLM manager first
        try:
            self.llm_manager = EnhancedLLMManager()
            logger.info("Using EnhancedLLMManager")
        except Exception as e:
            logger.warning(f"Failed to initialize Enhanced LLM Manager: {e}")
            try:
                self.llm_manager = SimpleLLMManager()
                logger.info("Falling back to SimpleLLMManager")
            except Exception as e2:
                logger.error(f"Could not initialize any LLM manager: {e2}")
                self.llm_manager = None
        
        # Replace category classifier with dynamic search processor
        self.dynamic_search = DynamicSearchProcessor(self.embedding_manager)
        
        # Initialize caches
        self.response_cache = TTLCache(maxsize=2000, ttl=3600)
        self.embedding_cache = TTLCache(maxsize=10000, ttl=3600)
        self.config_cache = TTLCache(maxsize=100, ttl=300)
        
        # For text matching with advanced tokenization
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
        
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
        """Extract search intent using the dynamic processor"""
        return self.dynamic_search.detect_intent(query)

    def _optimize_query(self, query: str, intent: Dict[str, Any]) -> str:
        """Optimize query based on detected intent"""
        # For high confidence category matches, explicitly include the category
        if intent["confidence"] > 0.7 and intent["category"] not in query.lower():
            optimized = f"{query} {intent['category']}"
            logger.info(f"Optimized query: {query} -> {optimized}")
            return optimized
            
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

            # 1. First stage: Detect intent using dynamic processor
            intent = self._detect_intent(query)
            logger.info(f"Detected intent: {intent}")
            
            # 2. Optimize query based on detected intent
            search_query = self._optimize_query(query, intent)
            
            # 3. Apply dynamic query expansion
            expanded_query = self.dynamic_search.expand_query(search_query, intent)

            # 4. Fetch relevant data using parallel processing
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
            
            # 5. Generate embedding for search query (use expanded query)
            query_embedding = self.embedding_manager.generate_embedding(
                [expanded_query], embedding_model
            )[0]

            # 6. Perform vector search with improved parameters
            search_start = time.time()
            vector_results = self.vector_store.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k * 3,  # Get more results for re-ranking with expanded query
                threshold=0.2,     # Lower threshold for better recall with expanded query
                filters=self._prepare_filters(filters, config.get("schema_mapping", {}), intent)
            )
            logger.info(f"Vector search completed in {time.time() - search_start:.3f}s")
            
            # 7. Implement fallback search strategies
            if not vector_results:
                logger.warning(f"No vector results found. Trying fallback search strategies.")
                
                # Try with original query
                if expanded_query != query:
                    original_query_embedding = self.embedding_manager.generate_embedding(
                        [query], embedding_model
                    )[0]
                    
                    vector_results = self.vector_store.search(
                        collection_name=collection_name,
                        query_vector=original_query_embedding.tolist(),
                        limit=top_k * 2,
                        threshold=0.15,  # Even lower threshold
                        filters=None  # Remove filters for maximum recall
                    )
                
                # If still no results, try generic category search
                if not vector_results and intent["category"]:
                    category_query = intent["category"]
                    logger.info(f"Trying category fallback with: {category_query}")
                    
                    category_embedding = self.embedding_manager.generate_embedding(
                        [category_query], embedding_model
                    )[0]
                    
                    vector_results = self.vector_store.search(
                        collection_name=collection_name,
                        query_vector=category_embedding.tolist(),
                        limit=top_k * 2,
                        threshold=0.1,  # Very low threshold for maximum recall
                        filters=None
                    )
            
            # Process empty results
            if not vector_results:
                # Check collection stats
                collection_stats = self.vector_store.get_collection_info(collection_name)
                vector_count = collection_stats.get("count", 0) if collection_stats else 0
                
                if (vector_count == 0):
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
                
                # Generate intelligent suggestion based on query
                suggestion = self._generate_search_suggestion(query, intent)
                
                return {
                    "results": [],
                    "generated_response": f"I couldn't find any products matching '{query}'. {suggestion}",
                    "search_metadata": {
                        "time_taken": time.time() - start_time,
                        "intent": intent,
                        "expanded_query": expanded_query,
                        "suggestion": suggestion
                    }
                }

            # Format results
            formatted_results = [
                self._format_search_result(result, config.get("schema_mapping", {}))
                for result in vector_results
            ]

            # Apply improved hybrid scoring
            texts = [
                f"{r['name']} {r['description']}" for r in formatted_results
            ]
            
            # Calculate similarity scores using different methods
            keyword_scores = self._compute_keyword_similarity(query, texts)
            semantic_scores = self._compute_semantic_similarity(query, texts, embedding_model)
            
            # Combine scores: 60% vector, 20% keyword, 20% semantic
            for idx, result in enumerate(formatted_results):
                if idx < len(keyword_scores):
                    hybrid_score = (
                        0.6 * result["score"] + 
                        0.2 * keyword_scores[idx] +
                        0.2 * semantic_scores[idx]
                    )
                    result["score"] = hybrid_score

            # Re-rank results with cross-encoder if available
            if self.cross_encoder and len(formatted_results) > 1:
                rerank_start = time.time()
                reranked_results = self._rerank_results(query, formatted_results)
                formatted_results = reranked_results[:top_k]
                logger.info(f"Reranking completed in {time.time() - rerank_start:.3f}s")
            else:
                # Sort by score and limit results
                formatted_results = sorted(
                    formatted_results, key=lambda x: x["score"], reverse=True
                )[:top_k]

            # Update dynamic search processor with actual results
            self.dynamic_search.detect_intent(query, formatted_results)

            # Generate response with LLM
            llm_start = time.time()
            generated_response = self._generate_product_recommendations(
                query, formatted_results, intent, expanded_query
            )
            logger.info(f"LLM response generated in {time.time() - llm_start:.3f}s")

            # Format final response
            response = {
                "generated_response": generated_response,
                "results": [self._format_result_for_frontend(r) for r in formatted_results],
                "search_metadata": {
                    "original_query": query,
                    "expanded_query": expanded_query,
                    "optimized_query": search_query,
                    "total_results": len(formatted_results),
                    "search_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat(),
                    "intent": intent
                }
            }

            # Additional validation before caching the response
            if "Required data fields" in generated_response or len(generated_response) < 20:
                # Try one more time with fallback generator
                if hasattr(self.llm_manager, "get_template_response"):
                    generated_response = self.llm_manager.get_template_response(
                        query, 
                        formatted_results[0] if formatted_results else None,
                        "jungle" if "jungle" in query else "outdoor"
                    )
                    response["generated_response"] = generated_response

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
            
    def _generate_search_suggestion(self, query: str, intent: Dict) -> str:
        """Generate intelligent search suggestions based on failed query"""
        # Start with default suggestion
        suggestion = "Please try using more general terms or check your spelling."
        
        # Generate category-specific suggestions
        if intent.get("category"):
            category = intent["category"]
            if "water" in query or "filter" in category or "purifier" in category:
                suggestion = "Try searching for 'water filter', 'water purifier', or 'water filtration system'."
            elif "kitchen" in category or "cook" in query:
                suggestion = "Try searching for kitchen appliances like 'mixer', 'blender', or 'cooking tools'."
            elif "clean" in query or "cleaning" in category:
                suggestion = "Try searching for 'cleaning supplies', 'vacuum cleaner', or 'mop'."
            elif "pest" in category or "mouse" in query or "insect" in query:
                suggestion = "Try searching for 'pest control', 'insect repellent', or 'mouse trap'."
            else:
                suggestion = f"Try searching for other {category} products with more general terms."
                 
        # Add some variety to suggestions based on query length
        if len(query.split()) > 4:
            suggestion += " You might also try a shorter, more specific query."
        else:
            suggestion += " You could also try adding more specific details to your search."
            
        return suggestion
             
    def _compute_semantic_similarity(self, query: str, texts: List[str], model_name: str) -> List[float]:
        """Compute semantic similarity between query and texts using embeddings"""
        try:
            # Generate embeddings
            query_embedding = self.embedding_manager.generate_embedding([query], model_name)[0]
            text_embeddings = self.embedding_manager.generate_embedding(texts, model_name)
            
            # Calculate cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            normalized_text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            
            # Calculate similarity scores
            scores = np.dot(normalized_text_embeddings, query_embedding)
            
            return scores.tolist()
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return [0.0] * len(texts)

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
                "name": result.get("name", ""),  # Include up to 5 products for better context
                "description": result.get("description", ""),
                "category": result.get("category", ""),
                "score": round(float(result.get("score", 0.0)), 4),
                **{k: str(v) for k, v in metadata.items()},
                "url": f"/product/{result.get('mongo_id', '')}"
            }
        except Exception as e:
            logger.error(f"Frontend formatting error: {e}")
            return {"error": "Formatting failed"}

    def _generate_product_recommendations(self, query: str, results: List[Dict], intent: Dict, expanded_query: str) -> str:
        """Generate tailored product recommendations with enhanced prompting"""
        if not results:
            return f"I couldn't find any products matching '{query}'."
            
        try:
            # Create product context with the most relevant information
            product_context = []
            for i, product in enumerate(results[:5]):  # Include up to 5 products for better context
                price = product.get("price", "N/A")
                if isinstance(price, str) and not price.startswith("₹") and not price.startswith("$"):
                    price = f"₹{price}"
                    
                # Clean up and format the product information
                name = product['name'][:80].strip()  # Limit name length
                description = product['description'][:100].strip()
                # Remove the [RELEVANT: ...] tags from description if present
                if "[RELEVANT:" in description:
                    description = description.split("[RELEVANT:")[0].strip()
                
                product_context.append(
                    f"Product {i+1}: {name}\n"
                    f"   Description: {description}\n"
                    f"   Price: {price}\n"
                    f"   Category: {product.get('category', 'N/A')}"
                )
                
            product_context_str = "\n\n".join(product_context)
            
            # Create structured intent information
            intent_info = []
            if intent.get("category"):
                intent_info.append(f"Product Category: {intent['category']}")
            if intent.get("product_features"):
                # Get top 5 most relevant features
                features = intent.get("product_features", [])[:5]
                intent_info.append(f"Key Features: {', '.join(features)}")
            if intent.get("outdoor_context"):
                intent_info.append("Context: Outdoor/Wilderness use")
                
            intent_str = "\n".join(intent_info)
            
            # Create specific instructions for the model to make it task-focused
            instructions = (
                "Based on the user's query, provide a helpful recommendation for the most suitable "
                "products. Focus on how these products match their specific needs for "
                f"{intent.get('category', 'this product category')}. Explain why they would be "
                "useful in the context mentioned (e.g., jungle, outdoor). Be concise and informative."
            )
            
            # Create prompt for LLM with improved structure and instructions
            prompt = (
                f"User Query: {query}\n\n"
                f"User Needs: {intent_str}\n\n"
                f"Available Products:\n{product_context_str}\n\n"
                f"Instructions: {instructions}\n\n"
                "Response:"
            )
            
            # Generate response with improved parameters
            response = self.llm_manager.generate_response(prompt, max_length=200)
            
            # Post-process the response to remove any irrelevant or garbled text
            if "https://" in response or "http://" in response:
                response = response.split("http")[0].strip()
            if "Required data" in response:
                # Fall back to a template response if generation is poor
                response = self._generate_fallback_response(query, results, intent)
                
            return response
                    
        except Exception as e:
            logger.error(f"Product recommendation generation failed: {e}")
            return self._generate_fallback_response(query, results, intent)
    
    def _generate_fallback_response(self, query: str, results: List[Dict], intent: Dict) -> str:
        """Generate a structured fallback response when LLM generation fails"""
        try:
            # Extract the top product
            top_product = results[0] if results else None
            
            if not top_product:
                return f"I couldn't find any products matching '{query}'."
                
            product_name = top_product.get('name', '').split('|')[0].strip()
            
            # Create a structured response based on the intent and query
            if "water" in query.lower() and "jungle" in query.lower():
                return (
                    f"For cleaning water in the jungle, I recommend the {product_name}. "
                    f"This product is designed to filter water in outdoor environments and would be suitable for your needs. "
                    f"It's portable and effective at removing contaminants to make water safe for drinking."
                )
            elif intent.get("category") == "water filter":
                return (
                    f"Based on your search for a water filter, I recommend the {product_name}. "
                    f"This product will help you clean water effectively and is suitable for your needs. "
                    f"It's one of our top-rated options in this category."
                )
            else:
                return (
                    f"Based on your search, I recommend the {product_name}. "
                    f"This product matches your requirements and is highly rated. "
                    f"Check out the details to see if it meets your specific needs."
                )
                
        except Exception:
            return "I found several products that might meet your needs. Please check the product list below for more details."

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

    def _compute_keyword_similarity(self, query: str, docs: List[str]) -> List[float]:
        """Compute improved keyword-based similarity for hybrid search"""
        try:
            # Extract query terms with stopword removal
            query_terms = set(term.lower() for term in query.split() 
                             if term.lower() not in self.dynamic_search.stopwords
                             and len(term) > 2)
                             
            if not query_terms:
                return [0.5] * len(docs)  # Neutral score if no meaningful terms
            
            # Calculate BM25-style scoring
            scores = []
            for doc in docs:
                if not doc:
                    scores.append(0.0)
                    continue
                    
                # Count term frequency in doc:
                doc_lower = doc.lower()
                term_matches = {}
                
                # Check for exact matches first (highest weight)
                exact_match_score = 0.0
                if query.lower() in doc_lower:
                    exact_match_score = 0.8
                    
                # Count individual term matches
                for term in query_terms:
                    count = doc_lower.count(term)
                    if count > 0:
                        # Weight by term length (longer terms more significant)
                        term_matches[term] = min(count, 3) * (len(term) / 10)
                        
                if not term_matches and exact_match_score == 0:
                    scores.append(0.0)
                    continue
                
                # Calculate score based on matched terms and their weights
                match_score = sum(term_matches.values()) / (len(query_terms) * 1.5)
                
                # Combine exact and term-based scores
                final_score = max(exact_match_score, min(match_score, 0.95))
                scores.append(final_score)
                
            return scores
        except Exception as e:
            logger.error(f"Keyword similarity calculation failed: {e}")
            return [0.0] * len(docs)

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



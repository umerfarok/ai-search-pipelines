"""
Dynamic search processor for intelligent query understanding
across multiple product categories and domains.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from cachetools import TTLCache
import nltk
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from domain_knowledge import (
    PRODUCT_CATEGORIES, 
    PRODUCT_FEATURES,
    DOMAIN_TERM_EXPANSION,
    QUERY_INTENTS,
    get_expanded_terms,
    get_related_categories,
    get_feature_terms
)

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)
 
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
        
        # Add vector embeddings for flexible concept matching
        self.concept_vectors = {}
        self._initialize_concept_vectors()
    
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
    
    def _initialize_concept_vectors(self):
        """Create concept vectors for semantic matching instead of keyword matching"""
        if not self.embedding_manager:
            logger.warning("No embedding manager available for concept vectors")
            return
            
        # Define general product concepts - these are not hardcoded categories
        # but rather semantic anchors for vector comparison
        concepts = {
            "water_treatment": [
                "clean drinking water", 
                "purify contaminated water",
                "filter out impurities from water",
                "make water safe to drink",
                "remove bacteria from water source"
            ],
            "portability": [
                "lightweight and easy to carry",
                "compact design for travel",
                "portable solution for on the go",
                "easy to transport equipment",
                "carry in backpack or bag"
            ],
            "outdoor_survival": [
                "equipment for wilderness survival",
                "tools for outdoor emergencies",
                "jungle survival gear",
                "forest expedition equipment",
                "outdoor adventure necessities"
            ],
            "kitchen_appliances": [
                "tools for food preparation",
                "cooking appliances for home",
                "kitchen equipment for meal making",
                "food processing devices",
                "appliances for heating food"
            ],
            "cleaning_products": [
                "cleaning solutions for surfaces",
                "products to remove dirt and stains",
                "disinfecting and sanitizing items",
                "household cleaning supplies",
                "cleaners for different surfaces"
            ],
            "pest_management": [
                "control insect populations",
                "repel mosquitoes and bugs",
                "eliminate pest infestations",
                "prevent bugs in home",
                "solutions for insect problems"
            ]
        }
        
        # Generate embeddings for each concept
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            for concept, examples in concepts.items():
                embeddings = self.embedding_manager.generate_embedding(examples, model_name)
                # Store the average embedding as the concept vector
                self.concept_vectors[concept] = np.mean(embeddings, axis=0)
                logger.info(f"Created vector for concept: {concept}")
        except Exception as e:
            logger.error(f"Failed to create concept vectors: {e}")

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
        """Detect intent using vector similarity rather than keyword matching"""
        # Check cache first
        if query in self.intent_cache:
            return self.intent_cache[query]
        
        # Basic intent features
        intent_data = {
            "category": None,
            "confidence": 0.0,
            "is_question": '?' in query,
            "keywords": [],
            "action": "find",
            "concepts": [],
            "vector_matches": []
        }
        
        # Extract keywords (excluding stopwords)
        words = query.lower().split()
        intent_data["keywords"] = [w for w in words if len(w) > 2 and w not in self.stopwords]
        
        # Instead of keyword matching, use vector similarity to find concepts
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.generate_embedding(
                [query], "sentence-transformers/all-MiniLM-L6-v2"
            )[0]
            
            # Find the most similar concepts
            similarities = []
            for concept, vector in self.concept_vectors.items():
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, vector) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(vector)
                )
                similarities.append((concept, float(similarity)))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Store top matches
            top_matches = [(concept, score) for concept, score in similarities if score > 0.3][:3]
            intent_data["vector_matches"] = top_matches
            
            # Set category based on best match
            if top_matches:
                best_concept, best_score = top_matches[0]
                intent_data["concepts"] = [concept for concept, _ in top_matches]
                
                # Map concept to product category
                category_mapping = {
                    "water_treatment": "water filter",
                    "outdoor_survival": "outdoor gear",
                    "kitchen_appliances": "kitchen appliance",
                    "cleaning_products": "cleaning supplies",
                    "pest_management": "pest control"
                }
                
                if best_concept in category_mapping:
                    intent_data["category"] = category_mapping[best_concept]
                    intent_data["confidence"] = best_score
                    intent_data["source"] = "vector_similarity"
        except Exception as e:
            logger.error(f"Error in vector-based intent detection: {e}")
            
        # If vector approach didn't work, fall back to existing methods
        if not intent_data["category"] and products:
            category = self._detect_category_from_products(query, products)
            if category:
                intent_data["category"] = category
                intent_data["confidence"] = 0.7
                intent_data["source"] = "product_analysis"
        
        # Cache the result
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
        """Expand query using vector similarity instead of hardcoded expansion rules"""
        # Check cache first
        if query in self.expansion_cache:
            return self.expansion_cache[query]
            
        # If we have vector matches from intent detection, use those to expand
        if intent and "concepts" in intent and intent["concepts"]:
            expanded_parts = [query]
            
            # Add expansion terms based on the matched concepts
            for concept in intent["concepts"][:2]:  # Use top two concepts
                if concept == "water_treatment":
                    expanded_parts.append("water purification filter clean drinking")
                elif concept == "portability":
                    expanded_parts.append("portable lightweight compact travel")
                elif concept == "outdoor_survival":
                    expanded_parts.append("wilderness jungle forest camping survival")
                elif concept == "kitchen_appliances":
                    expanded_parts.append("cooking food preparation kitchen appliance")
                elif concept == "cleaning_products":
                    expanded_parts.append("clean sanitize disinfect remove dirt")
                elif concept == "pest_management":
                    expanded_parts.append("insect bug repel control mosquito")
            
            expanded_query = " ".join(expanded_parts)
            self.expansion_cache[query] = expanded_query
            return expanded_query
            
        # If no vector matches, fall back to WordNet expansion
        # ...existing WordNet-based code...
        
        return query

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

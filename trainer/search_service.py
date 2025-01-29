import os
import logging
from flask import Flask, request, jsonify
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from pathlib import Path
import redis
from typing import Dict, List, Optional, Tuple
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import pandas as pd
import io
from dataclasses import dataclass
from datetime import datetime
import threading
from collections import OrderedDict
from enum import Enum
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

  
@dataclass
class AppConfig:
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    MODEL_CACHE_SIZE: int = int(os.getenv("MODEL_CACHE_SIZE", 2))
    MIN_SCORE: float = float(os.getenv("MIN_SCORE", 0.2))
    AWS_ACCESS_KEY: str = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY: str = os.getenv("AWS_SECRET_KEY")
    S3_BUCKET: str = os.getenv("S3_BUCKET")
    S3_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_SSL_VERIFY: bool = os.getenv("AWS_SSL_VERIFY", "true").lower() == "true"
    AWS_ENDPOINT_URL: str = os.getenv("AWS_ENDPOINT_URL")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    AVAILABLE_LLM_MODELS = {
        "deepseek-coder": {
            "name": "deepseek-ai/deepseek-coder-1.3b-base",
            "description": "Code-optimized model for technical content and embeddings"
        }
    }
    DEFAULT_MODEL = "deepseek-coder"
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "/tmp/model_cache")
    MODEL_CACHE_SIZE: int = int(os.getenv("MODEL_CACHE_SIZE", 2))
    
    @classmethod
    def get_cache_path(cls, model_path: str) -> str:
        """Get cached model path"""
        safe_path = model_path.replace("/", "_").replace("\\", "_")
        return os.path.join(cls.MODEL_CACHE_DIR, safe_path)


class S3Manager:
    def __init__(self):
        config = Config(retries=dict(max_attempts=3), s3={"addressing_style": "path"})

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AppConfig.AWS_ACCESS_KEY,
            aws_secret_access_key=AppConfig.AWS_SECRET_KEY,
            region_name=AppConfig.S3_REGION,
            endpoint_url=AppConfig.AWS_ENDPOINT_URL,
            verify=AppConfig.AWS_SSL_VERIFY,
            config=config,
        )
        self.bucket = AppConfig.S3_BUCKET
        logger.info(f"Initialized S3Manager with bucket: {self.bucket}")

    def download_file(self, s3_path: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3.download_file(self.bucket, s3_path, local_path)
            logger.info(f"Downloaded s3://{self.bucket}/{s3_path} to {local_path}")
            return True
        except ClientError as e:
            logger.error(f"Failed to download from S3: {e}")
            return False

    def get_csv_content(self, s3_path: str) -> Optional[pd.DataFrame]:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_path)
            return pd.read_csv(io.BytesIO(response["Body"].read()))
        except ClientError as e:
            logger.error(f"Failed to read CSV from S3: {e}")
            return None


class ModelCache:
    def __init__(self, cache_size: int = 2):
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        os.makedirs(AppConfig.MODEL_CACHE_DIR, exist_ok=True)

    def get_cache_path(self, model_path: str) -> str:
        return AppConfig.get_cache_path(model_path)
        
    def get(self, model_path: str) -> Optional[Dict]:
        with self.lock:
            cache_dir = self.get_cache_path(model_path)
            if os.path.exists(f"{cache_dir}/embeddings.npy"):
                if model_path in self.cache:
                    # Move to end (most recently used)
                    value = self.cache.pop(model_path)
                    self.cache[model_path] = value
                    return value
                    
                # Load from disk cache
                embeddings = np.load(f"{cache_dir}/embeddings.npy")
                with open(f"{cache_dir}/metadata.json", "r") as f:
                    metadata = json.load(f)
                    
                model_data = {
                    "embeddings": embeddings,
                    "metadata": metadata,
                    "loaded_at": datetime.now().isoformat(),
                }
                self.put(model_path, model_data)
                return model_data
            return None

    def put(self, model_path: str, model_data: Dict):
        with self.lock:
            cache_dir = self.get_cache_path(model_path)
            os.makedirs(cache_dir, exist_ok=True)
            
            # Save to disk cache
            np.save(f"{cache_dir}/embeddings.npy", model_data["embeddings"])
            with open(f"{cache_dir}/metadata.json", "w") as f:
                json.dump(model_data["metadata"], f)
            
            # Update memory cache
            if model_path in self.cache:
                self.cache.pop(model_path)
            elif len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[model_path] = model_data


class ProductLLMManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.s3_manager = S3Manager()
        self.model_info = None
        
    def load_model(self, model_path: str) -> bool:
        try:
            # Load model info first
            model_info_path = f"{model_path}/llm/model_info.json"
            if not os.path.exists(model_info_path):
                logger.warning(f"No LLM model info found at {model_info_path}")
                return False
                
            with open(model_info_path) as f:
                model_info = json.load(f)
                
            # Use the correct model based on saved info
            self.model_name = model_info.get("base_model")
            if not self.model_name:
                return False
                
            # Download all model files
            local_path = f"/tmp/models/{Path(model_path).name}/llm"
            os.makedirs(local_path, exist_ok=True)
            
            model_files = [
                "pytorch_model.bin", 
                "tokenizer_config.json",
                "config.json",
                "vocab.json",
                "special_tokens_map.json",
                "tokenizer.json"
            ]
            
            for file in model_files:
                s3_path = f"{model_path}/llm/{file}"
                local_file = f"{local_path}/{file}"
                if not self.s3_manager.download_file(s3_path, local_file):
                    logger.error(f"Failed to download {file}")
                    return False

            # Load the model and tokenizer
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            if torch.cuda.is_available():
                model_kwargs["load_in_4bit"] = True
                
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path,
                **model_kwargs
            )
            
            logger.info(f"Loaded fine-tuned model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            return False

    def generate_response(self, query: str, product_context: str) -> str:
        try:
            prompt = f"""Given the product information:
{product_context}

User query: {query}
Generate a natural response about the relevant products:"""

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("Generate a natural response about the relevant products:")[-1].strip()

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I couldn't generate a response about the products."


class QueryIntent(Enum):
    PRODUCT_SEARCH = "product_search"
    FOOD_RELATED = "food_related"
    PET_CARE = "pet_care"
    AGRICULTURE = "agriculture"
    PEST_CONTROL = "pest_control"
    CLEANING = "cleaning"
    SUPPORT = "support"
    GENERAL = "general"


class IntentClassifier:
    def __init__(self):
        """Initialize DeepSeek-Coder model for intent classification"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Use DeepSeek-Coder model which is smaller and focused on code/analysis
        model_name = "deepseek-ai/deepseek-coder-1.3b-base"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True,
                use_fast=True
            )
            
            # Configure model loading based on available hardware
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            # Only enable 4-bit quantization if GPU is available
            if torch.cuda.is_available():
                try:
                    import bitsandbytes as bnb
                    model_kwargs["load_in_4bit"] = True
                    model_kwargs["quantization_config"] = {
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": torch.float16
                    }
                except ImportError:
                    logger.warning("bitsandbytes not available, using standard precision")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            logger.info(f"Initialized model on {self.device} with kwargs: {model_kwargs}")

            # Define prompts for different intents
            self.intent_prompts = {
                "product_search": "Find products or items to purchase",
                "browsing": "Browse or explore categories",
                "support": "Customer service or help request",
                "information": "General information query"
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fallback to simpler model or raise error based on your needs
            raise

    def analyze_query(self, query: str) -> Dict:
        """Analyze query using DeepSeek-Coder to determine intent"""
        try:
            prompt = f"""Analyze this query and output JSON. Query: "{query}"
Instructions:
- Determine if this is a product search
- Identify user intent
- Suggest search keywords if product related
- Keep response focused and brief

Output format:
{{
    "is_product_search": true/false,
    "intent_type": "product_search/support/information",
    "keywords": ["keyword1", "keyword2"],
    "response": "Brief natural response"
}}
"""
            # Generate completion with smaller context
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON part using regex
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                # Fallback for invalid JSON
                result = {
                    "is_product_search": True,
                    "intent_type": "product_search",
                    "keywords": query.split(),
                    "response": "Let me search for relevant products."
                }

            return result

        except Exception as e:
            logger.error(f"Query analysis error: {e}")
            return {
                "is_product_search": True,
                "intent_type": "product_search",
                "keywords": query.split(),
                "response": "I'll search for products matching your query."
            }


class ResponseGenerator:
    def __init__(self):
        """Initialize response generator with dynamic intent handling"""
        self.default_response = {
            "prefix": "Here are some relevant products. ",
            "suggestions": []
        }
        
        # Load response templates from JSON file or database
        self.response_templates = self._load_response_templates()
    
    def _load_response_templates(self) -> Dict:
        """Load response templates from a configuration file"""
        # You can store these in a JSON file or database
        return {
            "food_cooking": {
                "prefix": "I'll help you find cooking and food-related items. ",
                "suggestions": ["kitchen appliances", "cookware", "ingredients"]
            },
            "pet_care": {
                "prefix": "Let me find pet care products for you. ",
                "suggestions": ["pet supplies", "grooming items", "pet food"]
            },
            # Add more templates as needed
        }
    
    def generate_response(self, intent: str, query: str, confidence: float) -> Dict:
        """Generate contextual response based on intent and confidence"""
        template = self.response_templates.get(intent, self.default_response)
        
        response = {
            "type": intent,
            "message": template["prefix"],
            "suggestions": template["suggestions"],
            "confidence": confidence,
            "original_query": query
        }
        
        # Add dynamic suggestions based on query context
        if confidence > 0.8:
            # You can add more sophisticated suggestion logic here
            response["enhanced"] = True
        
        return response


class SearchService:
    def __init__(self):
        self.s3_manager = S3Manager()
        self.model_cache = ModelCache(AppConfig.MODEL_CACHE_SIZE)
        self.device = AppConfig.DEVICE
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()
        self.llm_manager = ProductLLMManager()
        logger.info(f"Initialized SearchService using device: {self.device}")

    def load_model(self, model_path: str) -> Optional[Dict]:
        """Load model and its artifacts"""
        try:
            # Check cache first
            cached_model = self.model_cache.get(model_path)
            if cached_model:
                logger.info(f"Using cached model for {model_path}")
                return cached_model

            logger.info(f"Loading model from {model_path}")
            local_model_dir = f"/tmp/models/{Path(model_path).name}"
            os.makedirs(local_model_dir, exist_ok=True)

            # Download required files
            required_files = ["metadata.json", "embeddings.npy"]
            for file in required_files:
                s3_path = f"{model_path}/{file}"
                local_path = f"{local_model_dir}/{file}"
                if not self.s3_manager.download_file(s3_path, local_path):
                    raise FileNotFoundError(f"Failed to download {file}")

            # Load metadata and embeddings
            with open(f"{local_model_dir}/metadata.json", "r") as f:
                metadata = json.load(f)
            embeddings = np.load(f"{local_model_dir}/embeddings.npy")

            # Load LLM model
            llm_loaded = self.llm_manager.load_model(model_path)
            if not llm_loaded:
                raise ValueError("Failed to load LLM model")

            # Cache model data
            model_data = {
                "embeddings": embeddings,
                "metadata": metadata,
                "loaded_at": datetime.now().isoformat(),
            }

            self.model_cache.put(model_path, model_data)
            return model_data

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def search(
        self,
        query: str,
        model_path: str,
        max_items: int = 10,
        filters: Optional[Dict] = None,
    ) -> Dict:
        """Enhanced search with LLM-based intent analysis"""
        try:
            # First try to load fine-tuned LLM if not already loaded
            if not self.llm_manager.model:
                self.llm_manager.load_model(model_path)
            
            # Analyze query intent (use fine-tuned model if available)
            if self.llm_manager.model:
                analysis = self._analyze_with_finetuned_model(query)
            else:
                analysis = self.intent_classifier.analyze_query(query)
            
            # If it's not a product search, return conversational response
            if not analysis.get("is_product_search", True):
                return {
                    "intent": {
                        "type": analysis["intent_type"],
                        "is_product_search": False,
                        "response": analysis["response"]
                    },
                    "results": [],
                    "total": 0
                }
            
            # For product searches, use the existing search mechanism
            search_results = self._perform_search(
                analysis.get("keywords", query),
                model_path,
                max_items,
                filters
            )
            
            # Combine results with LLM response
            if search_results["results"]:
                # Create product context from top results
                product_context = self._create_product_context(search_results["results"][:3])
                
                # Generate natural language response
                natural_response = self.llm_manager.generate_response(query, product_context)
                
                # Add response to results
                search_results["natural_response"] = natural_response
            
            return search_results

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    def _perform_search(self, query: str, model_path: str, max_items: int, filters: Optional[Dict]) -> Dict:
        """Perform search using embeddings from LLM model"""
        try:
            model_data = self.load_model(model_path)
            if not model_data:
                raise ValueError(f"Failed to load model from {model_path}")

            # Generate query embedding using LLM model
            query_embedding = self._generate_embedding(query, self.llm_manager)
            embeddings = model_data["embeddings"]
            metadata = model_data["metadata"]

            # Calculate similarities
            similarities = np.dot(embeddings, query_embedding)
            top_k_idx = np.argsort(similarities)[-max_items:][::-1]

            # Load and filter products
            products_df = self.s3_manager.get_csv_content(f"{model_path}/products.csv")
            if products_df is None:
                raise FileNotFoundError("Products data not found")

            if filters:
                products_df = self.apply_filters(products_df, filters, metadata["config"]["schema_mapping"])

            # Prepare results
            results = []
            schema = metadata["config"]["schema_mapping"]

            for idx in top_k_idx:
                score = float(similarities[idx])
                if score < AppConfig.MIN_SCORE:
                    continue

                try:
                    product = products_df.iloc[idx]
                    result = {
                        "id": str(product.get(schema["id_column"], f"item_{idx}")),
                        "name": str(product.get(schema["name_column"], "Unknown")),
                        "description": str(product.get(schema["description_column"], "")),
                        "category": str(product.get(schema["category_column"], "Uncategorized")),
                        "score": score,
                        "metadata": {},
                    }

                    # Add custom column metadata
                    for col in schema.get("custom_columns", []):
                        if col["role"] == "metadata" and col["user_column"] in product:
                            result["metadata"][col["standard_column"]] = str(
                                product[col["user_column"]]
                            )

                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing result at index {idx}: {e}")
                    continue

            return {
                "results": results,
                "total": len(results),
                "query_info": {
                    "original": query,
                    "model_path": model_path,
                    "schema": schema,
                },
            }

        except Exception as e:
            logger.error(f"Search error: {e}")
            raise

    def apply_filters(
        self, df: pd.DataFrame, filters: Dict, schema_mapping: Dict
    ) -> pd.DataFrame:
        """Apply filters to the DataFrame"""
        filtered_df = df.copy()

        try:
            for field, value in filters.items():
                # Handle standard columns
                if field in ["category", "name", "description"]:
                    column = schema_mapping.get(f"{field}_column")
                    if column and column in filtered_df.columns:
                        filtered_df = filtered_df[
                            filtered_df[column].str.contains(
                                str(value), case=False, na=False
                            )
                        ]

                # Handle custom columns
                else:
                    for custom_col in schema_mapping.get("custom_columns", []):
                        if custom_col["standard_column"] == field:
                            column = custom_col["user_column"]
                            if column in filtered_df.columns:
                                if (
                                    isinstance(value, list) and len(value) == 2
                                ):  # Range filter
                                    filtered_df = filtered_df[
                                        (filtered_df[column] >= value[0])
                                        & (filtered_df[column] <= value[1])
                                    ]
                                else:  # Exact match or contains
                                    filtered_df = filtered_df[
                                        filtered_df[column]
                                        .astype(str)
                                        .str.contains(str(value), case=False, na=False)
                                    ]

        except Exception as e:
            logger.error(f"Error applying filters: {e}")

        return filtered_df

    def _create_product_context(self, products: List[Dict]) -> str:
        context = "Available products:\n\n"
        for idx, product in enumerate(products, 1):
            context += f"{idx}. {product['name']}\n"
            context += f"   Description: {product['description']}\n"
            context += f"   Category: {product['category']}\n"
            if product.get('metadata'):
                context += f"   Additional Info: {json.dumps(product['metadata'], indent=2)}\n"
            context += "\n"
        return context

    def _analyze_with_finetuned_model(self, query: str) -> Dict:
        try:
            prompt = f"""Analyze this product search query: "{query}"
Output format:
{{
    "is_product_search": true/false,
    "intent_type": "product_search/support/information",
    "keywords": ["keyword1", "keyword2"],
    "response": "Brief natural response"
}}
"""
            response = self.llm_manager.generate_response(prompt, "")
            # Parse JSON from response
            json_match = re.search(r'{.*}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return self.intent_classifier.analyze_query(query)  # Fallback
            
        except Exception as e:
            logger.error(f"Error analyzing with fine-tuned model: {e}")
            return self.intent_classifier.analyze_query(query)  # Fallback

    def _generate_embedding(self, text: str, model: 'ProductLLMManager') -> np.ndarray:
        """Generate embedding using LLM model"""
        try:
            inputs = model.tokenizer(text, return_tensors="pt", truncation=True, 
                                   max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.model(**inputs)
                # Use mean of last hidden states as embedding
                embedding = outputs.last_hidden_state.mean(dim=1)
                # Normalize embedding
                embedding = F.normalize(embedding, p=2, dim=1)
                
            return embedding.cpu().numpy()[0]

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise


# Initialize search service
search_service = SearchService()


@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json()
        if not data or "model_path" not in data or "query" not in data:
            return jsonify({"error": "Invalid request"}), 400

        model_path = data["model_path"]
        query = data["query"]
        max_items = data.get("max_items", 10)
        filters = data.get("filters", {})

        logger.info(f"Processing search request for model: {model_path}")        
        logger.info(f"Query: {query}")

        results = search_service.search(
            query=query, model_path=model_path, max_items=max_items, filters=filters
        )
        return jsonify(results)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Search error: {error_msg}")
        return jsonify({"error": error_msg}), 500


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "device": AppConfig.DEVICE,
            "cached_models": len(search_service.model_cache.cache),
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("SERVICE_PORT", 5000))
    app.run(host="0.0.0.0", port=port)

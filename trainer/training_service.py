import os
import logging
import time
from typing import Dict, List, Optional
import requests
import torch
import numpy as np
import json
from pathlib import Path
import redis
from datetime import datetime
import threading
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
from flask import Flask, jsonify
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import pandas as pd
import io
import re
from enum import Enum
from config import AppConfig
from vector_store import VectorStore  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class ModelStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ModelCache:
    def __init__(self, max_size: int = 4):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, model_path: str) -> Optional[Dict]:
        with self.lock:
            if model_path in self.cache:
                self.cache.move_to_end(model_path)
                return self.cache[model_path]
            return None

    def put(self, model_path: str, data: Dict):
        with self.lock:
            if (model_path in self.cache):
                self.cache.move_to_end(model_path)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
                self.cache[model_path] = data


class S3Manager:
    def __init__(self):
        retry_config = Config(
            retries=dict(max_attempts=AppConfig.MAX_RETRIES, mode="adaptive"),
            s3={"addressing_style": "path"},
        )

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AppConfig.AWS_ACCESS_KEY,
            aws_secret_access_key=AppConfig.AWS_SECRET_KEY,
            region_name=AppConfig.S3_REGION,
            endpoint_url=AppConfig.AWS_ENDPOINT_URL,
            verify=AppConfig.AWS_SSL_VERIFY.lower() == "true",
            config=retry_config,
        )
        self.bucket = AppConfig.S3_BUCKET

    def download_file(self, s3_path: str, local_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3.download_file(self.bucket, s3_path, local_path)
            return True
        except Exception as e:
            logger.error(f"Failed to download {s3_path}: {e}")
            return False

    def normalize_s3_path(self, path: str) -> str:
        return path.lstrip("/")

    def verify_file_exists(self, s3_path: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=s3_path)
            return True
        except ClientError:
            return False

    def get_csv_content(self, s3_path: str) -> Optional[pd.DataFrame]:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_path)
            content = response["Body"].read()
            if isinstance(content, bytes):
                csv_buffer = io.BytesIO(content)
            else:
                csv_buffer = io.BytesIO(content.encode("utf-8"))
            try:
                df = pd.read_csv(csv_buffer)
                logger.info(f"Successfully loaded CSV with {len(df)} rows")
                return df
            except pd.errors.EmptyDataError:
                logger.error("CSV file is empty")
                return None
            except pd.errors.ParserError as e:
                logger.error(f"Error parsing CSV: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Failed to read CSV from S3: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            return None

    def upload_file(self, local_path: str, s3_path: str) -> bool:
        try:
            content_type = {
                ".json": "application/json",
                ".npy": "application/octet-stream",
                ".csv": "text/csv",
            }.get(Path(local_path).suffix, "application/octet-stream")

            self.s3.upload_file(
                local_path,
                self.bucket,
                s3_path,
                ExtraArgs={"ContentType": content_type},
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False

    def normalize_path(self, path: str) -> str:
        """Normalize S3 path with smarter path handling"""
        path = path.lstrip("/")

        # Don't add models/ prefix for data files
        if path.startswith("data/"):
            return path

        # Add models/ prefix only for model-related files
        if not path.startswith("models/"):
            path = f"models/{path}"

        return path

    def get_csv_content(self, s3_path: str, retries: int = 3) -> Optional[pd.DataFrame]:
        """Enhanced CSV loading with proper path handling"""
        normalized_path = self.normalize_path(s3_path)
        logger.info(f"Attempting to load CSV from path: {s3_path}")
        logger.info(f"Normalized path: {normalized_path}")

        for attempt in range(retries):
            try:
                response = self.s3.get_object(Bucket=self.bucket, Key=normalized_path)
                content = response["Body"].read()

                if isinstance(content, bytes):
                    csv_buffer = io.BytesIO(content)
                else:
                    csv_buffer = io.BytesIO(content.encode("utf-8"))

                try:
                    df = pd.read_csv(csv_buffer)
                    logger.info(
                        f"Successfully loaded CSV with {len(df)} rows from {normalized_path}"
                    )
                    return df
                except pd.errors.EmptyDataError:
                    logger.error(f"CSV file is empty: {normalized_path}")
                    return None
                except pd.errors.ParserError as e:
                    logger.error(f"Error parsing CSV: {str(e)}")
                    return None

            except self.s3.exceptions.NoSuchKey:
                logger.warning(
                    f"File not found in S3: {normalized_path} (attempt {attempt + 1}/{retries})"
                )
                if attempt == retries - 1:
                    # Try without models/ prefix as last resort
                    try:
                        direct_path = path.lstrip("/")
                        response = self.s3.get_object(
                            Bucket=self.bucket, Key=direct_path
                        )
                        df = pd.read_csv(io.BytesIO(response["Body"].read()))
                        logger.info(
                            f"Successfully loaded CSV using direct path: {direct_path}"
                        )
                        return df
                    except Exception:
                        return None
            except Exception as e:
                logger.error(f"S3 error (attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt == retries - 1:
                    return None

            time.sleep(1 * (attempt + 1))

        return None

    def safe_upload_file(self, local_path: str, s3_path: str, retries: int = 3) -> bool:
        """Upload file with retries and verification"""
        normalized_path = self.normalize_path(s3_path)

        for attempt in range(retries):
            try:
                content_type = {
                    ".json": "application/json",
                    ".npy": "application/octet-stream",
                    ".csv": "text/csv",
                }.get(Path(local_path).suffix, "application/octet-stream")

                self.s3.upload_file(
                    local_path,
                    self.bucket,
                    normalized_path,
                    ExtraArgs={"ContentType": content_type},
                )

                # Verify upload
                if self.verify_file_exists(normalized_path):
                    logger.info(
                        f"Successfully uploaded and verified: {normalized_path}"
                    )
                    return True

            except Exception as e:
                logger.error(
                    f"Upload error (attempt {attempt + 1}/{retries}): {str(e)}"
                )
                if attempt == retries - 1:
                    return False

            time.sleep(1 * (attempt + 1))

        return False


class EmbeddingManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_models = {}
        self.model_load_lock = threading.Lock()
        self.tokenizers = {}
        self.max_seq_length = 512
        self.cache_dir = AppConfig.TRANSFORMER_CACHE

        # Exact model mappings matching frontend configuration
        self.MODEL_MAPPINGS = AppConfig.MODEL_MAPPINGS
        self.default_model_name = "all-minilm-l6"  # Matches frontend default

    def _get_model_info(self, model_name: str) -> dict:
        """Get model information with improved path matching"""
        # Check by key first
        if model_name in self.MODEL_MAPPINGS:
            return self.MODEL_MAPPINGS[model_name]

        # Check by path
        for key, info in self.MODEL_MAPPINGS.items():
            if info["path"] == model_name:
                return info

        # For direct paths that match our supported models
        for key, info in self.MODEL_MAPPINGS.items():
            if model_name.lower().replace("-", "") in info["path"].lower().replace(
                "-", ""
            ):
                return info

        raise ValueError(
            f"Unsupported model: {model_name}. Please use one of: {list(self.MODEL_MAPPINGS.keys())}"
        )

    def get_model(self, model_name: str = None):
        if not model_name:
            model_name = self.default_model_name

        model_info = self._get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unsupported model: {model_name}")

        model_path = model_info["path"]

        if model_path in self.embedding_models:
            return self.embedding_models[model_path]

        try:
            with self.model_load_lock:
                if model_path not in self.embedding_models:
                    logger.info(f"Loading model: {model_path}")
                    model = SentenceTransformer(
                        model_path, cache_folder=self.cache_dir, device=self.device
                    )

                    # Verify model dimension
                    actual_dim = model.get_sentence_embedding_dimension()
                    expected_dim = model_info["dimension"]
                    if actual_dim != expected_dim:
                        raise ValueError(
                            f"Model dimension mismatch for {model_name}: "
                            f"expected {expected_dim}, got {actual_dim}"
                        )

                    self.embedding_models[model_path] = model
                    logger.info(f"Successfully loaded model: {model_path}")
                return self.embedding_models[model_path]

        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {model_path}") from e

    def get_model_dimension(self, model_name: str) -> int:
        """Get the embedding dimension for a model"""
        model_info = self._get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Cannot determine dimension for model: {model_name}")
        return model_info["dimension"]

    def generate_embedding(self, text: List[str], model_name: str = None) -> np.ndarray:
        """Generate embeddings with improved error handling"""
        if not text:
            raise ValueError("Empty text input")

        model = self.get_model(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not initialized")

        # Dynamic batch size based on available memory
        batch_size = 512 if torch.cuda.is_available() else 128

        try:
            with torch.no_grad():
                embeddings = model.encode(
                    text,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                )

                # Verify embedding dimension
                expected_dim = self.get_model_dimension(model_name)
                if embeddings.shape[1] != expected_dim:
                    raise ValueError(
                        f"Generated embedding dimension {embeddings.shape[1]} "
                        f"doesn't match expected {expected_dim}"
                    )

                return embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if a model name is supported"""
        try:
            self._get_model_info(model_name)
            return True
        except ValueError:
            return False


class ProductTrainer:
    def __init__(self):
        self.default_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.vector_store = None  # Initialize as None
        self.embedding_manager = EmbeddingManager()
        self.s3_manager = S3Manager()
        self._id_counter = 0
        self._id_cache = {}

    def _process_text(self, row: pd.Series, schema: Dict) -> str:
        """Enhanced text processing with custom columns"""
        try:
            # Process core fields
            text_parts = [
                str(row[schema[f"{field}column"]])
                for field in ["name", "description", "category"]
                if schema.get(f"{field}column") in row
            ]

            # Process training custom columns
            if "customcolumns" in schema:
                for col in schema["customcolumns"]:
                    if col.get("role") == "training" and col["name"] in row:
                        text_parts.append(str(row[col["name"]]))

            return " ".join(filter(None, text_parts))

        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return " ".join(filter(None, text_parts[:3]))  # Return basic text if error

    def _load_data(self, config: Dict) -> Optional[pd.DataFrame]:
        """Enhanced data loading with better path handling and logging"""
        try:
            data_source = config["data_source"]
            current_file = data_source["location"]
            logger.info(f"Loading current data from: {current_file}")

            # Try to load the CSV file
            current_df = self.s3_manager.get_csv_content(current_file)
            if current_df is None:
                # Attempt with alternate path formats
                alternate_paths = [
                    current_file,
                    current_file.lstrip("/"),
                    f"data/{current_file}",
                    f"data/{current_file.lstrip('/')}",
                ]

                for path in alternate_paths:
                    logger.info(f"Attempting alternate path: {path}")
                    current_df = self.s3_manager.get_csv_content(path)
                    if current_df is not None:
                        break

            if current_df is None:
                raise ValueError(f"Failed to load current CSV from {current_file}")

            logger.info(
                f"Successfully loaded current data with shape: {current_df.shape}"
            )

            # Handle append mode
            if config.get("mode") == "append" and config.get("previous_version"):
                prev_path = f"models/{config['previous_version']}/products.csv"
                logger.info(f"Loading previous version from: {prev_path}")

                prev_df = self.s3_manager.get_csv_content(prev_path)
                if prev_df is not None:
                    logger.info(f"Loaded previous data with shape: {prev_df.shape}")

                    # Remove duplicates based on ID column if specified
                    if "idcolumn" in config["schema_mapping"]:
                        id_col = config["schema_mapping"]["idcolumn"]
                        combined_df = pd.concat(
                            [prev_df, current_df], ignore_index=True
                        )
                        combined_df = combined_df.drop_duplicates(
                            subset=[id_col], keep="last"
                        )
                        logger.info(
                            f"Combined shape after deduplication: {combined_df.shape}"
                        )
                    else:
                        combined_df = pd.concat(
                            [prev_df, current_df], ignore_index=True
                        )
                        logger.info(f"Combined shape: {combined_df.shape}")

                    return combined_df
                else:
                    logger.warning(
                        f"Previous version not found: {prev_path}, using only current data"
                    )
                    return current_df

            return current_df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise

    def _process_metadata(self, row: pd.Series, schema: Dict, config_id) -> Dict:
        try:
            mongo_id = str(config_id)
            return {
                "mongo_id": mongo_id,
                "id": str(row.get(schema["idcolumn"], "")),
                "name": str(row.get(schema["namecolumn"], "")),
                "description": str(row.get(schema["descriptioncolumn"], "")),
                "category": str(row.get(schema["categorycolumn"], "")),
                "custom_metadata": {
                    col["name"]: str(row[col["name"]])
                    for col in schema.get("customcolumns", [])
                    if col["name"] in row
                },
            }
        except Exception as e:
            logger.error(f"Metadata processing failed: {str(e)}")
            raise

    def train(self, job: Dict) -> bool:
        """Enhanced training with improved error handling and data access"""
        try:
            logger.info(f"Starting training with job: {str(job)[:100]}...")
            
            # Extract config data from job structure
            if isinstance(job, dict):
                if "config" in job:
                    config = job["config"]
                else:
                    config = job  # Job itself is the config
            
            if not config:
                raise ValueError("Invalid job structure - missing config")

            # Create ModelConfig from dictionary
            model_config = {
                "id": config.get("id"),
                "name": config.get("name"),
                "description": config.get("description", ""),
                "vector_size": config.get("vector_size", 384),  # Default size
                "embedding_model": config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                "index_config": {
                    "type": "hnsw",
                    "distance_metric": "cosine",
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 200,
                    "hnsw_ef": 100
                },
                "data_source": config.get("data_source", {}),
                "schema_mapping": config.get("schema_mapping", {})
            }

            logger.info(f"Processing config: {model_config['id']}")

            # Initialize vector store with explicit vector size
            vector_size = 384  # Default for all-MiniLM-L6-v2
            if model_config.get("vector_size"):
                vector_size = model_config["vector_size"]

            self.vector_store = VectorStore({
                "id": model_config["id"],
                "vector_size": vector_size,  # Make sure this is set
                "index_config": model_config.get("index_config", {
                    "type": "hnsw",
                    "distance_metric": "cosine",
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 200,
                    "hnsw_ef": 100
                })
            })

            # Process training
            schema = config["schema_mapping"]
            collection_name = f"products_{model_config['id']}"

            # Load and process data
            df = self._load_data({
                "data_source": config["data_source"],
                "mode": config.get("mode", "replace"),
                "previous_version": config.get("previous_version"),
                "schema_mapping": schema
            })

            if df is None or df.empty:
                raise ValueError("Empty dataset after loading")

            # Apply domain-specific enhancements to text based on product categories
            df = self._enhance_product_data(df, schema)
            
            # Generate better quality embeddings with domain awareness
            model_name = config["training_config"]["embeddingmodel"]
            expected_dim = self.embedding_manager.get_model_dimension(model_name)
            
            # Process text with improved contextual awareness
            texts = df.apply(lambda row: self._process_text(row, schema), axis=1).tolist()
            
            # Generate embeddings with potential chunking for very large datasets
            chunk_size = 1000  # Process in chunks to avoid memory issues
            all_embeddings = []
            
            for i in range(0, len(texts), chunk_size):
                chunk_texts = texts[i:i+chunk_size]
                chunk_embeddings = self.embedding_manager.generate_embedding(chunk_texts, model_name)
                all_embeddings.append(chunk_embeddings)
                
            embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

            # Create collection with enhanced metadata
            collection_name = f"products_{model_config['id']}"
            collection_metadata = {
                "config_id": model_config["id"],
                "embedding_model": model_name,
                "embedding_dimension": expected_dim,
                "schema_version": config.get("version", "1.0"),
                "record_count": len(df),
                "created_at": datetime.now().isoformat(),
                "data_source_columns": list(df.columns),
                "domain": config.get("domain", "general"),
                "schema_mapping": schema,
                "index_config": model_config["index_config"]
            }

            if not self.vector_store.create_collection(collection_name, collection_metadata):
                raise RuntimeError("Collection setup failed")

            # Process metadata with improved structure
            metadata_list = []
            for _, row in df.iterrows():
                try:
                    metadata = self._process_metadata(row, schema, model_config["id"])
                    # Ensure all text fields are properly normalized
                    for key, value in metadata.items():
                        if isinstance(value, str):
                            metadata[key] = value.strip()
                    metadata_list.append(metadata)
                except Exception as e:
                    logger.warning(f"Skipping invalid row: {str(e)}")
                    continue

            # Store vectors with optimized batching
            if not self.vector_store.upsert_vectors(
                collection_name=collection_name,
                embeddings=embeddings,
                metadata_list=metadata_list,
                ids=[str(i) for i in range(len(metadata_list))]
            ):
                raise RuntimeError("Vector storage failed")

            logger.info(f"Training completed successfully for config: {model_config['id']}")
            return True

        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            return False

    def _enhance_product_data(self, df, schema):
        """Enhance product data with domain-specific processing"""
        try:
            # Get column mappings
            name_col = schema.get("namecolumn")
            desc_col = schema.get("descriptioncolumn")
            cat_col = schema.get("categorycolumn")
            
            if not all([name_col, desc_col, cat_col]):
                return df  # Skip enhancement if required columns missing
                
            # Copy dataframe to avoid modifying original
            enhanced_df = df.copy()
            
            # Group products by category for domain-specific processing
            categories = df[cat_col].unique()
            
            for category in categories:
                # Apply category-specific enhancements
                cat_mask = enhanced_df[cat_col] == category
                cat_products = enhanced_df[cat_mask]
                
                if "water" in category.lower() or "filter" in category.lower():
                    # Enhance water-related product descriptions
                    enhanced_df.loc[cat_mask, desc_col] = cat_products[desc_col].apply(
                        lambda x: self._enhance_text_with_keywords(
                            x, 
                            ["clean", "purify", "filter", "portable", "drinking", "safe"]
                        )
                    )
                elif "outdoor" in category.lower() or "camping" in category.lower():
                    # Enhance outdoor product descriptions
                    enhanced_df.loc[cat_mask, desc_col] = cat_products[desc_col].apply(
                        lambda x: self._enhance_text_with_keywords(
                            x, 
                            ["portable", "durable", "lightweight", "compact", "survival"]
                        )
                    )
                    
            return enhanced_df
        except Exception as e:
            logger.warning(f"Error enhancing product data: {str(e)}")
            return df  # Return original if enhancement fails

    def _enhance_text_with_keywords(self, text, keywords):
        """Enhance text with relevant keywords if they're not already present"""
        if not text:
            return text
            
        text_lower = text.lower()
        enhancements = []
        
        for keyword in keywords:
            if keyword not in text_lower:
                # Check if semantically similar words are present
                # This is a simple implementation - could be improved with word embeddings
                if not any(similar in text_lower for similar in self._get_similar_words(keyword)):
                    enhancements.append(keyword)
                    
        if enhancements:
            # Add keywords as additional context without changing original text
            enhanced_text = f"{text} [Related: {', '.join(enhancements)}]"
            return enhanced_text
        
        return text

    def _get_similar_words(self, word):
        """Get similar words for a given keyword"""
        # This is a simple implementation - could be replaced with word embeddings
        similarity_map = AppConfig.DOMAIN_KEYWORDS
        return similarity_map.get(word, [])

    def _validate_embeddings(self, embeddings: np.ndarray, model_name: str) -> bool:
        """Enhanced embedding validation"""
        if not isinstance(embeddings, np.ndarray):
            logger.error("Embeddings not numpy array")
            return False

        if embeddings.dtype != np.float32:
            logger.error("Embeddings not float32")
            return False

        expected_dim = self.embedding_manager.get_model_dimension(model_name)
        if embeddings.shape[1] != expected_dim:
            logger.error(f"Dimension mismatch: {embeddings.shape[1]} vs {expected_dim}")
            return False

        return True

    def _save_model_files(
        self,
        model_dir: str,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        config: Dict,
        processed_texts: List[str],
        embedding_dimension: int,
    ) -> bool:
        """Enhanced model file saving with additional metadata and validation"""
        try:
            os.makedirs(model_dir, exist_ok=True)

            # Save embeddings and validate
            embeddings_path = os.path.join(model_dir, "embeddings.npy")
            np.save(embeddings_path, embeddings)

            # Verify saved embeddings
            loaded_embeddings = np.load(embeddings_path)
            if not np.array_equal(embeddings, loaded_embeddings):
                raise ValueError("Embeddings verification failed after saving")

            # Save products data
            df.to_csv(os.path.join(model_dir, "products.csv"), index=False)

            # Save processed texts for BM25
            with open(os.path.join(model_dir, "processed_texts.json"), "w") as f:
                json.dump(processed_texts, f)

            # Enhanced metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "num_samples": len(df),
                "embedding_shape": embeddings.shape,
                "models": {"embeddings": config["training_config"]["embedding_model"]},
                "schema_mapping": config["schema_mapping"],
                "training_stats": {
                    "start_time": datetime.now().isoformat(),
                    "processed_records": len(df),
                    "total_records": len(df),
                    "embedding_dimension": embedding_dimension,
                    "embedding_model": config["training_config"]["embedding_model"],
                    "batch_size": config["training_config"].get("batchsize", 128),
                    "max_tokens": config["training_config"].get("maxtokens", 512),
                },
                "version": {
                    "model": config.get("version", "unknown"),
                    "created_at": datetime.now().isoformat(),
                },
                "data_source": config["data_source"],  # Save data source info
            }

            with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Error saving model files: {e}")
            return False

    def _save_embeddings(self, model_dir: str, embeddings: np.ndarray) -> bool:
        """Save embeddings with validation"""
        try:
            # Save embeddings
            embeddings_path = os.path.join(model_dir, "embeddings.npy")
            np.save(embeddings_path, embeddings)

            # Verify saved embeddings
            loaded_embeddings = np.load(embeddings_path)
            if not np.array_equal(embeddings, loaded_embeddings):
                raise ValueError("Embedding verification failed after saving")

            return True
        except Exception as e:
            logger.error(f"Error saving embeddings: e")
            return False

    def _upload_model_files(self, local_dir: str, model_path: str) -> bool:
        try:
            files = [
                "embeddings.npy",
                "metadata.json",
                "products.csv",
                "processed_texts.json",
            ]

            for file in files:
                local_path = os.path.join(local_dir, file)
                s3_path = f"{model_path}/{file}"

                if not os.path.exists(local_path):
                    raise ValueError(f"Required file missing: {file}")

                if not self.s3_manager.upload_file(local_path, s3_path):
                    raise Exception(f"Failed to upload {file}")

            return True

        except Exception as e:
            logger.error(f"Error uploading files: {e}")
            return False


class TrainingWorker:
    def __init__(self):
        self.trainer = ProductTrainer()
        self.redis = redis.Redis(
            host=AppConfig.REDIS_HOST,
            port=AppConfig.REDIS_PORT,
            password=AppConfig.REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=5,
            retry_on_timeout=True,
            socket_keepalive=True,
        )
        self.should_stop = False
        self.current_job = None
        self._worker_thread = None

    def process_job(self, job: Dict):
        """Process a single training job"""
        try:
            logger.info(f"Processing job for config: {job.get('config_id')}")
            config = None

            # Extract config properly from job structure
            if isinstance(job, dict):
                if "config" in job:
                    config = job["config"]
                elif all(k in job for k in ["id", "name", "schema_mapping"]):
                    config = job  # Job itself is the config

            if not config:
                raise ValueError("Invalid job structure - missing config")

            # Set default values for required fields
            config.setdefault("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            config.setdefault("vector_size", 384)  # Default for all-MiniLM-L6-v2
            
            # Set default index config if missing
            if not config.get("index_config") or not config["index_config"].get("type"):
                config["index_config"] = {
                    "type": "hnsw",
                    "distance_metric": "cosine",
                    "hnsw_m": 16,
                    "hnsw_ef_construction": 200,
                    "hnsw_ef": 100
                }

            # Update the training configuration
            config.setdefault("training_config", {})
            config["training_config"].update({
                "model_type": "transformer",
                "embeddingmodel": config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                "batch_size": 128,
                "max_tokens": 512,
                "validation_split": 0.2
            })

            # Ensure data source format is set
            if "data_source" in config and isinstance(config["data_source"], dict):
                config["data_source"].setdefault("format", "csv")
                if not config["data_source"].get("options"):
                    config["data_source"]["options"] = {}

            config_id = config.get("id")
            if not config_id:
                raise ValueError("Config ID is required")

            logger.info(f"Processing config: {config_id}")
            self._update_status(config_id, ModelStatus.PROCESSING, progress=0)
            self.update_api_status(config_id, ModelStatus.PROCESSING.value, progress=0)

            # Use schema mapping directly without conversion
            success = self.trainer.train({
                "config": config,
                "config_id": config_id
            })

            if success:
                self._update_status(config_id, ModelStatus.COMPLETED, progress=100)
                self.update_api_status(config_id, ModelStatus.COMPLETED.value, progress=100)
            else:
                self._update_status(
                    config_id, ModelStatus.FAILED, error="Training failed"
                )
                self.update_api_status(config_id, ModelStatus.FAILED.value, error="Training failed")

        except Exception as e:
            logger.error(f"Error processing job: {e}")
            config_id = job.get("config_id") or (job.get("config", {}).get("id"))
            if config_id:
                self._update_status(
                    config_id, ModelStatus.FAILED, error=str(e)
                )
                self.update_api_status(config_id, ModelStatus.FAILED.value, error=str(e))
            else:
                logger.error("Could not update status: invalid job structure")

    def _validate_schema_mapping(self, config: Dict) -> bool:
        """Validate schema mapping configuration"""
        required_fields = [
            "idcolumn",
            "namecolumn",
            "descriptioncolumn",
            "categorycolumn",
        ]
        schema = config.get("schema_mapping", {})

        # Check required fields
        if not all(field in schema for field in required_fields):
            return False

        # Validate custom columns if present
        if "customcolumns" in schema:
            for col in schema["customcolumns"]:
                if not all(field in col for field in ["name", "type", "role"]):
                    return False

        return True

    def update_api_status(
        self,
        config_id: str,
        status: str,
        progress: float = None,
        error: str = None,
        model_info: dict = None,
    ):
        """Update status via API with retries"""
        try:
            update_data = {"status": status, "updated_at": datetime.now().isoformat()}
            if progress is not None:
                update_data["progress"] = progress
            if error:
                update_data["error"] = error
            if model_info:
                update_data.update(model_info)

            url = f"{AppConfig.API_HOST}/config/status/{config_id}"

            for attempt in range(AppConfig.MAX_RETRIES):
                try:
                    response = requests.put(url, json=update_data, timeout=5)
                    if response.status_code == 200:
                        return
                    if attempt < AppConfig.MAX_RETRIES - 1:
                        time.sleep(AppConfig.RETRY_DELAY * (attempt + 1))
                except Exception as e:
                    if attempt == AppConfig.MAX_RETRIES - 1:
                        logger.error(f"Error updating status via API: {e}")
                    else:
                        time.sleep(AppConfig.RETRY_DELAY * (attempt + 1))

        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def _update_status(
        self,
        config_id: str,
        status: ModelStatus,
        progress: float = None,
        error: str = None,
    ):
        """Update job status in Redis"""
        try:
            status_data = {
                "status": status.value,
                "timestamp": datetime.now().isoformat(),
            }

            if progress is not None:
                status_data["training_stats"] = {
                    "progress": progress,
                    "processed_records": 0,
                    "total_records": 0,
                    "start_time": datetime.now().isoformat() if progress == 0 else None,
                    "end_time": datetime.now().isoformat() if progress == 100 else None,
                }
            if error:
                status_data["error"] = error

            key = f"{AppConfig.MODEL_STATUS_PREFIX}{config_id}"
            self.redis.set(key, json.dumps(status_data), ex=86400)  # 24 hour expiry
            self.update_api_status(config_id, status.value, progress, error)

        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def _worker_loop(self):
        """Main worker loop"""
        logger.info("Starting training worker loop")
        while not self.should_stop:
            try:
                result = self.redis.brpop(AppConfig.TRAINING_QUEUE, timeout=1)
                print(result)
                if result:
                    _, job_data = result
                    job = json.loads(job_data)
                    self.current_job = job
                    self.process_job(job)
                    self.current_job = None
                else:
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1)

    def start(self):
        """Start the training worker"""
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.should_stop = False
            self._worker_thread = threading.Thread(target=self._worker_loop)
            self._worker_thread.daemon = True
            self._worker_thread.start()
            logger.info("Training worker started")

    def stop(self):
        """Stop the training worker"""
        logger.info("Stopping training worker")
        self.should_stop = True
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
            logger.info("Training worker stopped")


worker = TrainingWorker()


@app.route("/models")
def get_available_models():
    """Get available embedding models info"""
    return jsonify(
        {
            "models": AppConfig.MODEL_MAPPINGS,
            "default": AppConfig.DEFAULT_MODEL,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )


@app.route("/health")
def health():
    """Health check endpoint"""
    try:
        redis_ok = worker.redis.ping()
        return jsonify(
            {
                "status": "healthy",
                "redis": redis_ok,
                "worker": {
                    "running": worker._worker_thread is not None
                    and worker._worker_thread.is_alive(),
                    "current_job": (
                        worker.current_job["_id"]["$oid"]
                        if worker.current_job
                        else None
                    ),
                },
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
        )
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


def initialize_service():
    """Initialize all service components"""
    logger.info("Initializing service...")

    # Start worker
    worker.start()

    logger.info("Service initialization completed")
    return True


if __name__ == "__main__":
    print("Starting Train service...")
    if not initialize_service():
        logger.error("Service initialization failed")
    app.run(host="0.0.0.0", port=AppConfig.SERVICE_PORT, use_reloader=False)

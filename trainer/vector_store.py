import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import uuid

# Change from relative import to absolute import
from config import ModelConfig, AppConfig, IndexConfig, DistanceMetric
from qdrant_client.http import models  # Add this import at the top

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Interface to various vector databases.
    Currently supports Qdrant and ChromaDB.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize VectorStore with dictionary config"""
        if config is None:
            config = {}
        
        self.config = config
        self.collection_name = f"model_{config.get('id')}" if config.get('id') else None
        self.vector_size = config.get('vector_size', 384)  # Default size
        self._client = None
        self._id_counter = 0
        self._connection_retries = 3
        self._retry_delay = 2  # seconds

    def _generate_uuid(self, base: str = None) -> str:
        """Generate a valid UUID for Qdrant"""
        if base:
            # Create a UUID5 using base string
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))
        return str(uuid.uuid4())

    def create_collection(self, collection_name: Optional[str] = None, metadata: Optional[Dict] = None) -> bool:
        """Create vector collection with metadata."""
        try:
            collection_name = collection_name or self.collection_name
            if not collection_name:
                raise ValueError("Collection name is required")

            # Handle Qdrant
            if hasattr(AppConfig, 'QDRANT_HOST') and AppConfig.QDRANT_HOST:
                client = self._get_qdrant_client()
                from qdrant_client.http import models
                from qdrant_client.http.models import VectorParams, Distance

                # Delete existing collection if it exists
                try:
                    client.delete_collection(collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
                except Exception:
                    pass

                # Create vector params properly with correct structure
                vector_params = VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )

                # Create new collection with vector params
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=vector_params,
                )
                logger.info(f"Created collection: {collection_name}")

                # Add metadata as a separate point if provided
                if metadata:
                    try:
                        metadata_uuid = self._generate_uuid("metadata")
                        metadata_point = models.PointStruct(
                            id=metadata_uuid,
                            vector=np.zeros(self.vector_size).tolist(),  # Zero vector
                            payload={
                                "type": "collection_metadata",
                                "metadata": metadata,
                                "created_at": datetime.now().isoformat()
                            }
                        )
                        client.upsert(
                            collection_name=collection_name,
                            points=[metadata_point]
                        )
                        logger.info(f"Added metadata to collection: {collection_name}")
                    except Exception as e:
                        logger.warning(f"Failed to set collection metadata: {e}")

            # Handle ChromaDB
            elif hasattr(AppConfig, 'VECTOR_DB_HOST') and AppConfig.VECTOR_DB_HOST:
                client = self._get_chroma_client()
                try:
                    client.delete_collection(collection_name)
                except:
                    pass
                
                client.create_collection(
                    name=collection_name,
                    metadata=metadata or {"dimension": self.vector_size}
                )
            
            return True
        except Exception as e:
            logger.error(f"Error creating vector collection: {e}")
            if hasattr(e, 'content'):
                logger.error(f"Raw response content:\n{e.content.decode()}")
            return False

    def upsert_vectors(self, collection_name: str, embeddings: np.ndarray, 
                      metadata_list: List[Dict], ids: Optional[List[str]] = None) -> bool:
        """Upsert vectors to collection."""
        try:
            if ids is None:
                ids = [str(i) for i in range(len(embeddings))]
            
            # Convert numpy array to list of lists
            vectors = embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings

            # Handle Qdrant
            if hasattr(AppConfig, 'QDRANT_HOST') and AppConfig.QDRANT_HOST:
                client = self._get_qdrant_client()
                from qdrant_client.http import models

                # Create points with proper UUID generation
                points = []
                for idx, (vector, meta) in enumerate(zip(embeddings, metadata_list)):
                    # Generate a UUID based on metadata id or index
                    point_id = self._generate_uuid(meta.get('id', str(idx)))
                    
                    # Convert vector to list if needed
                    vector_data = vector.tolist() if isinstance(vector, np.ndarray) else vector
                    
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector=vector_data,
                            payload=meta
                        )
                    )

                # Batch upload points
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    client.upsert(
                        collection_name=collection_name,
                        points=batch,
                    )
            
            # Handle ChromaDB
            elif hasattr(AppConfig, 'VECTOR_DB_HOST') and AppConfig.VECTOR_DB_HOST:
                client = self._get_chroma_client()
                collection = client.get_collection(collection_name)
                
                collection.add(
                    ids=ids,
                    embeddings=vectors,
                    metadatas=metadata_list
                )
            
            return True
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            return False

    def _get_qdrant_client(self):
        """Get or create Qdrant client with retry logic."""
        if self._client:
            return self._client

        from qdrant_client import QdrantClient
        
        for attempt in range(self._connection_retries):
            try:
                logger.info(f"Connecting to Qdrant at {AppConfig.QDRANT_HOST}:{AppConfig.QDRANT_PORT}")
                client = QdrantClient(
                    host=AppConfig.QDRANT_HOST,
                    port=AppConfig.QDRANT_PORT,
                    timeout=30,  # Increased timeout
                    prefer_grpc=True,
                )
                
                # Test connection
                client.get_collections()
                logger.info("Successfully connected to Qdrant")
                self._client = client
                return client
                
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant (attempt {attempt + 1}/{self._connection_retries}): {e}")
                if attempt == self._connection_retries - 1:
                    raise
                time.sleep(self._retry_delay * (attempt + 1))

    def _get_chroma_client(self):
        """Get or create ChromaDB client."""
        try:
            import chromadb
            
            # Initialize client only once
            if self._client is None:
                self._client = chromadb.HttpClient(
                    host=AppConfig.VECTOR_DB_HOST,
                    port=AppConfig.VECTOR_DB_PORT
                )
            return self._client
        except ImportError:
            raise ImportError("ChromaDB client not installed. Please install it with `pip install chromadb`.")
    
    def add_vectors(self, ids: List[str], vectors: List[List[float]], 
                    metadata: List[Dict[str, Any]]) -> bool:
        """
        Add vectors to the collection with their metadata.
        """
        try:
            # Handle Qdrant
            if hasattr(AppConfig, 'QDRANT_HOST') and AppConfig.QDRANT_HOST:
                client = self._get_qdrant_client()
                from qdrant_client.http import models
                
                points = [
                    models.PointStruct(
                        id=id_str,
                        vector=vector,
                        payload=meta
                    )
                    for id_str, vector, meta in zip(ids, vectors, metadata)
                ]
                
                client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            # Handle ChromaDB
            elif hasattr(AppConfig, 'VECTOR_DB_HOST') and AppConfig.VECTOR_DB_HOST:
                client = self._get_chroma_client()
                collection = client.get_collection(self.collection_name)
                
                collection.add(
                    ids=ids,
                    embeddings=vectors,
                    metadatas=metadata
                )
            
            return True
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            return False
    
    def search(self, query_vector: List[float] = None, collection_name: str = None, 
               top_k: int = 10, limit: int = 10, threshold: float = 0.0, 
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search vectors similar to the query vector with updated parameter handling.
        
        Args:
            query_vector: Vector to search for
            collection_name: Name of collection to search in (overrides self.collection_name)
            top_k: Number of results to return (alias for limit)
            limit: Number of results to return
            threshold: Minimum similarity score
            filters: Filtering conditions
            
        Returns:
            List of search results with metadata
        """
        try:
            # Handle collection name properly
            coll_name = collection_name or self.collection_name
            if not coll_name:
                raise ValueError("Collection name is required")
                
            # Handle top_k/limit parameter
            result_limit = limit if limit else top_k
            
            results = []
            
            # Handle Qdrant
            if hasattr(AppConfig, 'QDRANT_HOST') and AppConfig.QDRANT_HOST:
                client = self._get_qdrant_client()
                
                # Build filter if provided
                filter_query = None
                if filters:
                    from qdrant_client.http import models
                    conditions = []
                    for key, value in filters.items():
                        if isinstance(value, list):
                            conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchAny(any=value)
                                )
                            )
                        else:
                            conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchValue(value=value)
                                )
                            )
                    
                    filter_query = models.Filter(
                        must=conditions
                    )
                
                search_results = client.search(
                    collection_name=coll_name,
                    query_vector=query_vector,
                    limit=result_limit,
                    score_threshold=threshold,
                    query_filter=filter_query
                )
                
                results = [
                    {
                        "id": str(hit.id),
                        "score": hit.score,
                        "metadata": hit.payload
                    }
                    for hit in search_results
                ]
            
            # Handle ChromaDB
            elif hasattr(AppConfig, 'VECTOR_DB_HOST') and AppConfig.VECTOR_DB_HOST:
                client = self._get_chroma_client()
                collection = client.get_collection(coll_name)
                
                # Convert filter format if needed
                chroma_where = filters if filters else None
                
                search_results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=result_limit,
                    where=chroma_where
                )
                
                if search_results:
                    for i in range(len(search_results["ids"][0])):
                        results.append({
                            "id": search_results["ids"][0][i],
                            "score": search_results["distances"][0][i],
                            "metadata": search_results["metadatas"][0][i]
                        })
            
            return results
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def delete_collection(self) -> bool:
        """Delete the vector collection."""
        try:
            # Handle Qdrant
            if hasattr(AppConfig, 'QDRANT_HOST') and AppConfig.QDRANT_HOST:
                client = self._get_qdrant_client()
                client.delete_collection(collection_name=self.collection_name)
            
            # Handle ChromaDB
            elif hasattr(AppConfig, 'VECTOR_DB_HOST') and AppConfig.VECTOR_DB_HOST:
                client = self._get_chroma_client()
                client.delete_collection(self.collection_name)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = {
                "name": self.collection_name,
                "vector_size": self.vector_size,
                "count": 0
            }
            
            # Handle Qdrant
            if hasattr(AppConfig, 'QDRANT_HOST') and AppConfig.QDRANT_HOST:
                client = self._get_qdrant_client()
                collection_info = client.get_collection(collection_name=self.collection_name)
                info["count"] = collection_info.vectors_count
                info["config"] = {
                    "distance": collection_info.config.params.vectors.distance,
                    "size": collection_info.config.params.vectors.size
                }
            
            # Handle ChromaDB
            elif hasattr(AppConfig, 'VECTOR_DB_HOST') and AppConfig.VECTOR_DB_HOST:
                client = self._get_chroma_client()
                collection = client.get_collection(self.collection_name)
                info["count"] = collection.count()
            
            return info
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"name": self.collection_name, "error": str(e)}

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists with improved error handling."""
        try:
            if hasattr(AppConfig, 'QDRANT_HOST') and AppConfig.QDRANT_HOST:
                client = self._get_qdrant_client()
                if not client:
                    logger.error("No Qdrant client available")
                    return False
                    
                collections = client.get_collections().collections
                return any(c.name == collection_name for c in collections)
                
            # Handle ChromaDB
            elif hasattr(AppConfig, 'VECTOR_DB_HOST') and AppConfig.VECTOR_DB_HOST:
                client = self._get_chroma_client()
                collections = client.list_collections()
                return any(c.name == collection_name for c in collections)
            
            return False
        except Exception as e:
            logger.error(f"Error checking collection existence: {e}")
            return False

    def get_collection_metadata(self, collection_name: str) -> Optional[Dict]:
        """Get metadata for a collection with improved metadata handling."""
        try:
            if hasattr(AppConfig, 'QDRANT_HOST') and AppConfig.QDRANT_HOST:
                client = self._get_qdrant_client()
                
                try:
                    # First try: Get metadata from collection info
                    collection_info = client.get_collection(collection_name)
                    collection_metadata = getattr(collection_info, 'payload', None)
                    if collection_metadata:
                        return collection_metadata
                    
                    # Second try: Search for metadata point
                    metadata_uuid = self._generate_uuid("metadata")
                    search_result = client.scroll(
                        collection_name=collection_name,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="type",
                                    match=models.MatchValue(value="collection_metadata")
                                )
                            ]
                        ),
                        limit=1
                    )
                    
                    if search_result and search_result[0]:
                        points = search_result[0]
                        if points and points[0].payload:
                            return points[0].payload.get("metadata", {})
                    
                    # Third try: Search all points for metadata
                    scroll_result = client.scroll(
                        collection_name=collection_name,
                        with_payload=True,
                        limit=1
                    )
                    
                    if scroll_result and scroll_result[0]:
                        for point in scroll_result[0]:
                            if point.payload and "metadata" in point.payload:
                                return point.payload["metadata"]
                    
                    logger.warning(f"No metadata found for collection {collection_name}")
                    return {}
                    
                except Exception as e:
                    logger.error(f"Error retrieving Qdrant metadata: {e}")
                    return {}
                
            # Handle ChromaDB
            elif hasattr(AppConfig, 'VECTOR_DB_HOST') and AppConfig.VECTOR_DB_HOST:
                client = self._get_chroma_client()
                collection = client.get_collection(collection_name)
                return collection.metadata or {}
            
            return {}
        except Exception as e:
            logger.error(f"Error getting collection metadata: {e}")
            return {}

def create_vector_store(config: dict, documents: list) -> None:
    """Create a vector store from documents based on config."""
    # Ensure we're using the correct field names from updated ModelConfig
    model_path = config.get('ModelPath')
    embedding_model = config.get('EmbeddingModel', 'sentence-transformers/all-MiniLM-L6-v2')
    
    # Extract vector store configuration
    vector_config = {
        "id": config.get("id"),
        "vector_size": config.get("vector_size", 384),  # Default for all-MiniLM-L6-v2
        "index_config": {
            "type": "hnsw",
            "distance_metric": "cosine",
            "hnsw_m": 16,
            "hnsw_ef_construction": 200,
            "hnsw_ef": 100
        }
    }

    try:
        # Initialize vector store
        vector_store = VectorStore(vector_config)
        collection_name = f"products_{config['id']}"

        # Prepare metadata
        metadata = {
            "config_id": config["id"],
            "embedding_model": embedding_model,
            "embedding_dimension": vector_config["vector_size"],
            "schema_version": config.get("version", "1.0"),
            "record_count": len(documents),
            "created_at": datetime.now().isoformat(),
            "schema_mapping": config.get("schema_mapping", {}),
            "index_config": vector_config["index_config"]
        }

        # Create collection
        if not vector_store.create_collection(collection_name, metadata):
            raise RuntimeError("Failed to create collection")

        # Process documents and metadata
        metadata_list = []
        for i, doc in enumerate(documents):
            metadata_list.append({
                "id": str(doc.get("id", i)),
                "name": doc.get("name", ""),
                "description": doc.get("description", ""),
                "category": doc.get("category", ""),
                "custom_metadata": doc.get("metadata", {})
            })

        # Generate embeddings in batches
        embedding_manager = EmbeddingManager()
        batch_size = 1000
        all_embeddings = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.get("text", "") for doc in batch]
            batch_embeddings = embedding_manager.generate_embedding(texts, embedding_model)
            all_embeddings.append(batch_embeddings)

        # Combine all embeddings
        embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

        # Upload vectors to collection
        if not vector_store.upsert_vectors(
            collection_name=collection_name,
            embeddings=embeddings,
            metadata_list=metadata_list,
            ids=[str(i) for i in range(len(metadata_list))]
        ):
            raise RuntimeError("Failed to upload vectors")

        logger.info(f"Successfully created vector store for collection: {collection_name}")
        return True

    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return False

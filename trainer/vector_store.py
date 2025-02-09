import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional
from datetime import datetime
import logging
import numpy as np
import uuid

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "qdrant"), port=6333, prefer_grpc=True
        )

    def create_collection(self, name: str, metadata: Dict) -> bool:
        """Create collection with metadata cleanup"""
        try:
            # Delete existing collection if it exists
            if self.collection_exists(name):
                # First remove any existing metadata points
                self.client.delete(
                    collection_name=name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="type",
                                    match=models.MatchValue(
                                        value="collection_metadata"
                                    ),
                                )
                            ]
                        )
                    ),
                )
                self.client.delete_collection(name)
                logger.info(f"Cleaned up existing collection: {name}")

            # Create new collection
            self.client.recreate_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=metadata["embedding_dimension"],
                    distance=models.Distance.COSINE,
                ),
            )

            # Add single metadata point
            metadata_point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.0] * metadata["embedding_dimension"],
                payload={
                    "metadata": metadata,
                    "type": "collection_metadata",
                    "created_at": datetime.now().isoformat(),
                },
            )
            self.client.upsert(collection_name=name, points=[metadata_point], wait=True)

            logger.info(f"Created collection {name} with metadata")
            return True

        except Exception as e:
            logger.error(f"Collection creation failed: {str(e)}")
            return False

    def get_collection_metadata(self, collection_name: str) -> Optional[Dict]:
        """Retrieve collection metadata"""
        try:
            # Search for metadata points
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="type",
                            match=models.MatchValue(value="collection_metadata"),
                        )
                    ]
                ),
                limit=1,
            )

            if results and len(results[0]) > 0:
                return results[0][0].payload.get("metadata")
            return None
        except Exception as e:
            logger.error(f"Metadata retrieval failed: {str(e)}")
            return None

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception as e:
            return False

    def update_collection_metadata(self, collection_name: str, metadata: Dict) -> bool:
        """Update collection metadata"""
        try:
            # Create new metadata point
            metadata_point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=[0.0] * metadata["embedding_dimension"],
                payload={
                    "metadata": metadata,
                    "type": "collection_metadata",
                    "updated_at": datetime.now().isoformat(),
                },
            )

            # Remove old metadata
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="type",
                                match=models.MatchValue(value="collection_metadata"),
                            )
                        ]
                    )
                ),
            )

            # Insert new metadata
            self.client.upsert(
                collection_name=collection_name, points=[metadata_point], wait=True
            )

            return True
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")
            return False

    def upsert_vectors(
        self,
        collection_name: str,
        embeddings: np.ndarray,
        metadata_list: List[Dict],
        ids: List[str],
    ) -> bool:
        """Store vectors using UUIDs for Qdrant IDs and MongoDB IDs in payload"""
        try:
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),  # Generate new UUID for Qdrant
                    vector=embedding.tolist(),
                    payload=metadata,
                )
                for embedding, metadata in zip(embeddings, metadata_list)
            ]

            # Batch insert with 500 items per batch
            batch_size = 500
            for i in range(0, len(points), batch_size):
                self.client.upsert(
                    collection_name=collection_name,
                    points=points[i : i + batch_size],
                    wait=True,
                )

            logger.info(f"Successfully inserted {len(points)} vectors")
            return True

        except Exception as e:
            logger.error(f"Vector storage failed: {str(e)}")
            return False

    def search(
        self, collection_name: str, query_vector: List[float], limit: int = 10
    ) -> List[Dict]:
        """Search vectors with relevance scores"""
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )

            return [
                {"id": result.id, "score": result.score, "metadata": result.payload}
                for result in results
            ]
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def get_collection_size(self, collection_name: str) -> int:
        """Get accurate collection count with metadata awareness"""
        try:
            response = self.client.count(
                collection_name=collection_name, exact=True, count_filter=None
            )
            return response.count
        except Exception as e:
            logger.error(f"Collection count failed: {str(e)}")
            return 0

    def get_metadata_count(self, collection_name: str) -> int:
        results = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="type", match=models.MatchValue(value="collection_metadata")
                    )
                ]
            ),
        )
        return len(results[0])

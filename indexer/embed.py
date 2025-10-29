"""
Embedding Generation with Namespace Versioning
===============================================

Generate embeddings for chunks with:
- Model namespace versioning
- Batch processing
- Device optimization (CPU/CUDA/MPS)
- Idempotent operations

Usage:
    embedder = Embedder(model_namespace="bge-m3@1024@v1")
    embeddings = embedder.embed_chunks(chunks)
"""

from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Embedding result"""
    chunk_id: str
    embedding: np.ndarray
    model_namespace: str
    dimension: int


class Embedder:
    """Generate embeddings with namespace versioning"""

    def __init__(
        self,
        model_namespace: str = "bge-m3@1024@v1",
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        device: str = "auto"
    ):
        """
        Initialize embedder

        Args:
            model_namespace: Namespace identifier (model@dim@version)
            model_name: Actual model name/path
            batch_size: Batch size for embedding
            device: Device (auto, cpu, cuda, mps)
        """
        self.model_namespace = model_namespace
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        # Parse namespace
        parts = model_namespace.split('@')
        if len(parts) >= 2:
            self.expected_dimension = int(parts[1])
        else:
            self.expected_dimension = None

        # Initialize embedding model (import existing embedder)
        self._init_model()

    def _init_model(self):
        """Initialize embedding model"""
        try:
            from src.embedding.advanced_embedding import EmbeddingManager, EmbeddingConfig

            # Create config
            config = EmbeddingConfig(
                model_name=self.model_name,
                model_type="sentence_transformers",
                device=self.device,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                cache_enabled=True
            )

            # Initialize manager
            self.embed_manager = EmbeddingManager(config)
            self.embed_manager.initialize()

            # Get model info
            model_info = self.embed_manager.get_model_info()
            actual_dimension = model_info.get('dimension', 0)

            print(f"✅ Initialized embedder: {self.model_namespace}")
            print(f"   Model: {self.model_name}")
            print(f"   Device: {config.device}")
            print(f"   Dimension: {actual_dimension}")

            # Verify dimension matches namespace
            if self.expected_dimension and actual_dimension != self.expected_dimension:
                print(f"⚠️  Warning: Model dimension ({actual_dimension}) "
                      f"doesn't match namespace ({self.expected_dimension})")

        except Exception as e:
            print(f"❌ Failed to initialize embedder: {e}")
            raise

    def embed_chunks(self, chunks: List) -> List[EmbeddingResult]:
        """
        Generate embeddings for chunks

        Args:
            chunks: List of Chunk objects

        Returns:
            List of EmbeddingResult objects
        """
        results = []

        # Extract texts
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_chunks = chunks[i:i + self.batch_size]

            # Generate embeddings
            embeddings = self.embed_manager.encode(batch_texts)

            # Create results
            for chunk, embedding in zip(batch_chunks, embeddings):
                results.append(EmbeddingResult(
                    chunk_id=chunk.chunk_id,
                    embedding=embedding,
                    model_namespace=self.model_namespace,
                    dimension=len(embedding)
                ))

        return results

    def get_namespace(self) -> str:
        """Get model namespace"""
        return self.model_namespace


# Example usage
if __name__ == "__main__":
    print("Embedder module ready")

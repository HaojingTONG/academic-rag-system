"""
Upsert and Delete Operations for Vector DB
==========================================

Idempotent upsert and delete operations with:
- Primary key: (doc_id, chunk_id, embed_model_ns)
- Batch upsert
- Soft/hard delete
- Namespace-aware operations

Usage:
    upserter = VectorDBUpserter(collection_name="papers_v1")
    upserter.upsert_chunks(chunks, embeddings, namespace)
    upserter.delete_document(doc_id, namespace)
"""

from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings


class VectorDBUpserter:
    """Manage upsert and delete operations"""

    def __init__(
        self,
        collection_name: str = "papers_v1",
        db_path: str = "vector_db",
        distance: str = "cosine"
    ):
        """
        Initialize upserter

        Args:
            collection_name: Collection name
            db_path: Path to ChromaDB
            distance: Distance metric
        """
        self.collection_name = collection_name
        self.db_path = db_path
        self.distance = distance

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name
            )
            print(f"✅ Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance}
            )
            print(f"✅ Created new collection: {collection_name}")

    def upsert_chunks(
        self,
        chunks: List,
        embeddings: List,
        namespace: str,
        batch_size: int = 100
    ):
        """
        Upsert chunks to vector DB (idempotent)

        Args:
            chunks: List of Chunk objects
            embeddings: List of EmbeddingResult objects
            namespace: Embedding model namespace
            batch_size: Batch size for upsert
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        total = len(chunks)

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            # Prepare data
            ids = []
            embeddings_list = []
            metadatas = []
            documents = []

            for chunk, emb in zip(batch_chunks, batch_embeddings):
                # Primary key: {doc_id}::{chunk_id}::{namespace}
                pk = f"{chunk.doc_id}::{chunk.chunk_id}::{namespace}"
                ids.append(pk)

                # Embedding
                embeddings_list.append(emb.embedding.tolist())

                # Metadata
                metadata = {
                    'doc_id': chunk.doc_id,
                    'chunk_id': chunk.chunk_id,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'embed_model_ns': namespace,
                    'title': chunk.title or '',
                    'section': chunk.section or '',
                    'tokens': chunk.tokens
                }

                # Add custom metadata if present
                if chunk.metadata:
                    if 'year' in chunk.metadata:
                        metadata['year'] = chunk.metadata['year']
                    if 'authors' in chunk.metadata:
                        metadata['authors'] = chunk.metadata['authors']
                    if 'doi' in chunk.metadata:
                        metadata['doi'] = chunk.metadata['doi']

                metadatas.append(metadata)

                # Document text
                documents.append(chunk.text)

            # Upsert (idempotent - updates if exists, inserts if new)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents
            )

            print(f"   Upserted batch {i//batch_size + 1}: {len(ids)} chunks")

        print(f"✅ Upserted {total} chunks to {self.collection_name}")

    def delete_document(
        self,
        doc_id: str,
        namespace: Optional[str] = None,
        hard: bool = False
    ):
        """
        Delete all chunks for a document

        Args:
            doc_id: Document ID
            namespace: Optional namespace filter
            hard: If True, hard delete; else soft delete (not supported in ChromaDB, always hard)
        """
        # Build where filter
        where = {'doc_id': doc_id}
        if namespace:
            where['embed_model_ns'] = namespace

        # Get matching IDs
        results = self.collection.get(
            where=where,
            include=[]
        )

        if results['ids']:
            # Delete
            self.collection.delete(ids=results['ids'])
            print(f"✅ Deleted {len(results['ids'])} chunks for doc_id={doc_id}")
        else:
            print(f"⚠️  No chunks found for doc_id={doc_id}")

    def delete_namespace(self, namespace: str):
        """Delete all chunks for a namespace"""
        results = self.collection.get(
            where={'embed_model_ns': namespace},
            include=[]
        )

        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"✅ Deleted {len(results['ids'])} chunks for namespace={namespace}")
        else:
            print(f"⚠️  No chunks found for namespace={namespace}")

    def count_documents(self, namespace: Optional[str] = None) -> int:
        """Count unique documents in collection"""
        where = {'embed_model_ns': namespace} if namespace else None

        results = self.collection.get(
            where=where,
            include=['metadatas']
        )

        # Count unique doc_ids
        doc_ids = set(meta['doc_id'] for meta in results['metadatas'])
        return len(doc_ids)

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        all_results = self.collection.get(include=['metadatas'])

        total_chunks = len(all_results['ids'])
        doc_ids = set(meta['doc_id'] for meta in all_results['metadatas'])
        namespaces = set(meta.get('embed_model_ns', 'unknown') for meta in all_results['metadatas'])

        # Count by namespace
        namespace_counts = {}
        for meta in all_results['metadatas']:
            ns = meta.get('embed_model_ns', 'unknown')
            namespace_counts[ns] = namespace_counts.get(ns, 0) + 1

        return {
            'total_chunks': total_chunks,
            'unique_documents': len(doc_ids),
            'namespaces': list(namespaces),
            'namespace_counts': namespace_counts
        }


# Example usage
if __name__ == "__main__":
    upserter = VectorDBUpserter(collection_name="papers_test")

    # Get stats
    stats = upserter.get_stats()
    print(f"Stats: {stats}")

#!/usr/bin/env python3
"""
Build BM25 Index from Vector Store
===================================

This script builds a BM25 index from the documents already in the vector store.
The BM25 index enables keyword-based retrieval to complement dense vector search.

Usage:
    python build_bm25_index.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pickle
from typing import List, Dict
from rag.retriever import VectorStore, BM25Retriever

def build_bm25_index(output_path: str = "data/bm25_index.pkl"):
    """
    Build BM25 index from vector store documents

    Args:
        output_path: Path to save the BM25 index
    """
    print("🚀 Building BM25 Index from Vector Store...")
    print("=" * 70)

    # Initialize vector store
    print("\n📦 [1/4] Loading Vector Store...")
    vector_store = VectorStore()
    stats = vector_store.get_stats()

    print(f"   ✓ Loaded {stats['total_documents']} documents")
    print(f"   ✓ Embedding model: {stats['embedding_model']}")

    # Get all documents from vector store
    print("\n📄 [2/4] Extracting documents...")
    collection = vector_store.collection

    # Get all documents
    results = collection.get(
        include=['documents', 'metadatas']
    )

    documents = results['documents']
    metadatas = results['metadatas']
    doc_ids = results['ids']

    print(f"   ✓ Extracted {len(documents)} documents")
    print(f"   ✓ Average length: {sum(len(d) for d in documents) / len(documents):.0f} chars")

    # Build BM25 index
    print("\n🔨 [3/4] Building BM25 index...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)

    print("   - Tokenizing documents...")
    bm25.fit(
        documents=documents,
        doc_ids=doc_ids,
        metadatas=metadatas
    )

    print(f"   ✓ BM25 index built")
    print(f"   ✓ Vocabulary size: {len(bm25.idf)} unique terms")

    # Save index
    print(f"\n💾 [4/4] Saving index to {output_path}...")

    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(bm25, f)

    # Check file size
    file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
    print(f"   ✓ Saved ({file_size:.2f} MB)")

    # Test retrieval
    print("\n✅ [TEST] Testing BM25 retrieval...")
    test_query = "transformer attention mechanism"
    test_results = bm25.search(test_query, top_k=3)

    print(f"   Query: \"{test_query}\"")
    print(f"   Results: {len(test_results)}")
    if test_results:
        print(f"   Top score: {test_results[0].bm25_score:.4f}")
        print(f"   Top doc preview: {test_results[0].content[:100]}...")

    print("\n" + "=" * 70)
    print("✅ BM25 Index built successfully!")
    print(f"📁 Saved to: {output_path}")
    print("\nNext steps:")
    print("  1. The index will be automatically loaded by the RAG pipeline")
    print("  2. Hybrid retrieval (Dense + BM25) is now enabled")
    print("  3. Run queries to see improved retrieval quality")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BM25 index from vector store")
    parser.add_argument('--output', type=str, default="data/bm25_index.pkl",
                       help="Output path for BM25 index")
    args = parser.parse_args()

    build_bm25_index(args.output)

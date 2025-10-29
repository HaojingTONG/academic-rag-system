#!/usr/bin/env python3
"""
Test Hybrid Retrieval (Dense + BM25)
=====================================

Quick test to verify hybrid retrieval is working.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rag.pipeline import RAGPipeline, RAGConfig

def test_hybrid_retrieval():
    """Test that hybrid retrieval is working"""
    print("🧪 Testing Hybrid Retrieval (Dense + BM25)")
    print("=" * 70)

    # Initialize pipeline with hybrid retrieval enabled
    config = RAGConfig(
        use_hybrid_retrieval=True,
        vector_weight=0.7,
        bm25_weight=0.3,
        top_k=5
    )

    pipeline = RAGPipeline(config)
    pipeline.initialize()

    # Test query
    test_query = "What is the transformer architecture?"

    print(f"\n📝 Test Query: {test_query}")
    print("=" * 70)

    result = pipeline.query(
        question=test_query,
        top_k=5,
        return_metadata=True
    )

    # Check results
    print("\n📊 Results:")
    print(f"   ✓ Retrieved: {result['num_sources']} sources")
    print(f"   ✓ Success: {result['success']}")

    if result.get('metadata'):
        metadata = result['metadata']
        print(f"\n⏱️  Performance:")
        print(f"   - Retrieval: {metadata['retrieval_time']:.3f}s")
        print(f"   - Ranking: {metadata['ranking_time']:.3f}s")
        print(f"   - Generation: {metadata['generation_time']:.3f}s")
        print(f"   - Total: {metadata['total_time']:.3f}s")

    print("\n📄 Sources:")
    for i, source in enumerate(result['sources'][:3], 1):
        title = source.get('title', 'Unknown')
        score = source.get('score', 0)
        print(f"   [{i}] {title} (score: {score:.4f})")

    print("\n" + "=" * 70)

    # Check if hybrid retrieval was used
    status = pipeline.get_status()
    if hasattr(pipeline.retriever, 'bm25_retriever'):
        print("✅ Hybrid Retrieval is ENABLED")
        print(f"   - Vector weight: {pipeline.retriever.vector_weight}")
        print(f"   - BM25 weight: {pipeline.retriever.bm25_weight}")
    else:
        print("⚠️  Vector-only retrieval (BM25 not available)")

    print("=" * 70)


if __name__ == "__main__":
    test_hybrid_retrieval()

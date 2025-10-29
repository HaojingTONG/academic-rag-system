#!/usr/bin/env python3
"""
Test BGE Reranker
==================

Quick test to verify bge-reranker-large is working.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rag.pipeline import RAGPipeline, RAGConfig

def test_bge_reranker():
    """Test that BGE reranker is working"""
    print("🧪 Testing BGE Reranker (bge-reranker-large)")
    print("=" * 70)

    # Initialize pipeline with reranking enabled
    config = RAGConfig(
        use_hybrid_retrieval=True,
        enable_reranking=True,  # This should load bge-reranker-large
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

    print("\n📄 Top Sources (after reranking):")
    for i, source in enumerate(result['sources'][:5], 1):
        title = source.get('title', 'Unknown')
        score = source.get('score', 0)
        print(f"   [{i}] {title}")
        print(f"       Score: {score:.4f}")

    print("\n" + "=" * 70)

    # Check if BGE reranker was used
    if hasattr(pipeline.ranker, 'reranker'):
        reranker = pipeline.ranker.reranker
        if reranker.model_type == "bge":
            print("✅ BGE Reranker (bge-reranker-large) is ENABLED")
        elif reranker.model_type == "cross-encoder":
            print("⚠️  Using fallback cross-encoder (not BGE)")
        else:
            print("⚠️  Using feature-based reranking (models not available)")
    else:
        print("⚠️  Reranking is disabled")

    print("=" * 70)


if __name__ == "__main__":
    test_bge_reranker()

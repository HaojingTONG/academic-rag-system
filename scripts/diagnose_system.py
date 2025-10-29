#!/usr/bin/env python3
"""
RAG System Diagnostic Tool
==========================
Analyzes current system state to identify retrieval and generation issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def check_vector_store():
    """Check vector store stats"""
    print_section("1. Vector Store Diagnostics")

    try:
        from rag.retriever import VectorStore

        store = VectorStore()

        # Get collection
        collection = store.collection

        # Count vectors
        count = collection.count()
        print(f"✓ Total vectors in collection: {count}")

        # Sample a few documents to check metadata
        if count > 0:
            sample = collection.peek(limit=3)
            print(f"\n✓ Sample documents:")
            for i, (doc_id, metadata) in enumerate(zip(sample['ids'], sample['metadatas']), 1):
                title = metadata.get('title', 'N/A')[:60]
                print(f"  [{i}] {doc_id}: {title}...")

            # Check embedding dimensions
            if 'embeddings' in sample and sample['embeddings']:
                dim = len(sample['embeddings'][0])
                print(f"\n✓ Embedding dimension: {dim}")

                # Check norm
                import numpy as np
                norms = [np.linalg.norm(emb) for emb in sample['embeddings'][:5]]
                avg_norm = np.mean(norms)
                print(f"✓ Average embedding norm: {avg_norm:.3f}")
                print(f"  (Normalized: {0.95 < avg_norm < 1.05})")

        return True
    except Exception as e:
        print(f"✗ Error checking vector store: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_raw_retrieval():
    """Test retrieval without filters"""
    print_section("2. Raw Retrieval Test (No Filters)")

    try:
        from rag.retriever import VectorStore

        store = VectorStore()

        # Test query
        query = "What is Transformer Architecture?"
        print(f"Query: '{query}'")

        # Raw search - no filters
        print(f"\nSearching (top_k=20, no threshold)...")
        results = store.search(query, top_k=20, similarity_threshold=0.0)

        print(f"\n✓ Retrieved {len(results)} results")

        if len(results) > 0:
            print(f"\nTop 5 results:")
            for i, result in enumerate(results[:5], 1):
                score = result.vector_score
                title = result.metadata.get('title', 'Unknown')[:60]
                content_preview = result.content[:80].replace('\n', ' ')
                print(f"  [{i}] Score: {score:.4f} | {title}")
                print(f"      Preview: {content_preview}...")
        else:
            print("✗ No results retrieved!")

        return len(results) > 0
    except Exception as e:
        print(f"✗ Error in retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_filtered_retrieval():
    """Test retrieval with current filters"""
    print_section("3. Filtered Retrieval Test (Current Config)")

    try:
        from configs import config
        from rag.retriever import VectorStore
        from rag.ranker import CompositeRanker

        store = VectorStore()

        # Create ranker with current config
        ranker = CompositeRanker(
            enable_reranking=config.retrieval.enable_rerank,
            enable_filtering=True,
            enable_diversification=config.retrieval.enable_mmr,
            enable_deduplication=True
        )

        query = "What is Transformer Architecture?"
        print(f"Query: '{query}'")
        print(f"Config threshold: {config.retrieval.similarity_threshold}")

        # Retrieve
        results = store.search(query, top_k=20)
        print(f"\n✓ Raw retrieval: {len(results)} results")

        # Apply filters
        filtered = ranker.process(query, results, top_k=5)
        print(f"✓ After filtering: {len(filtered)} results")

        if len(filtered) == 0:
            print("\n⚠️ PROBLEM: All results filtered out!")
            print("   Analyzing why...")

            # Check score distribution
            if results:
                scores = [r.vector_score for r in results]
                print(f"\n   Score distribution:")
                print(f"   - Max: {max(scores):.4f}")
                print(f"   - Min: {min(scores):.4f}")
                print(f"   - Avg: {sum(scores)/len(scores):.4f}")
                print(f"   - Threshold: {config.retrieval.similarity_threshold}")

                below_threshold = sum(1 for s in scores if s < config.retrieval.similarity_threshold)
                print(f"\n   {below_threshold}/{len(scores)} results below threshold!")

        return len(filtered) > 0
    except Exception as e:
        print(f"✗ Error in filtered retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation():
    """Test LLM generation"""
    print_section("4. Generation Test")

    try:
        from src.generator.llm_client import create_llm_manager_from_config

        manager = create_llm_manager_from_config()

        print(f"Backend: {manager.backend}")
        print(f"Model: {manager.preferred_model}")
        print(f"Available: {manager.openai_available if manager.backend == 'openai' else manager.ollama_available}")

        # Test prompt
        test_prompt = """Answer this question based on the context:

Context: The Transformer architecture uses self-attention mechanisms.

Question: What is the Transformer?

Answer:"""

        print(f"\nTesting generation with sample prompt...")
        response = manager.generate_answer(test_prompt, max_tokens=100)

        print(f"\n✓ Success: {response.success}")
        print(f"✓ Model used: {response.model}")
        print(f"✓ Response preview: {response.text[:200]}...")

        # Check if it's fallback
        if 'fallback' in response.model.lower() or '基于相关学术文献' in response.text:
            print(f"\n⚠️ WARNING: Using fallback generator!")
            print(f"   This indicates LLM backend is not working")

        return response.success
    except Exception as e:
        print(f"✗ Error in generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all diagnostics"""
    print("\n" + "="*70)
    print("  RAG System Diagnostic Report")
    print("="*70)

    results = {}

    # Run tests
    results['vector_store'] = check_vector_store()
    results['raw_retrieval'] = test_raw_retrieval()
    results['filtered_retrieval'] = test_filtered_retrieval()
    results['generation'] = test_generation()

    # Summary
    print_section("Diagnostic Summary")

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")

    passed = sum(results.values())
    total = len(results)

    print(f"\n  Score: {passed}/{total} tests passed")

    if passed < total:
        print(f"\n  🚨 CRITICAL ISSUES FOUND")
        print(f"     System requires immediate fixes")
    else:
        print(f"\n  ✅ All tests passed")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())

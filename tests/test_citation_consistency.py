#!/usr/bin/env python3
"""
Test Citation Consistency and Generation Improvements
=====================================================

Tests the optimized RAG system with:
- New prompt composer
- Generator with retry logic and multi-provider fallback
- Citation sanitization (only show cited sources)
- Full content injection in prompts
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from rag.pipeline import RAGPipeline, RAGConfig
from rag.citation_utils import get_cited_numbers, CitationFormatter
import json


def test_transformer_query():
    """Test with 'What is Transformer Architecture?' query"""
    print("=" * 80)
    print("  Testing Citation Consistency with Transformer Architecture Query")
    print("=" * 80)

    # Initialize pipeline
    config = RAGConfig(
        use_hybrid_retrieval=True,
        enable_reranking=True,
        top_k=5
    )

    pipeline = RAGPipeline(config)
    pipeline.initialize()

    # Test query
    test_query = "What is Transformer Architecture?"

    print(f"\n📝 Test Query: {test_query}")
    print("=" * 80)

    # Execute query
    result = pipeline.query(
        question=test_query,
        top_k=5,
        return_metadata=True
    )

    # Analyze results
    print("\n" + "=" * 80)
    print("  RESULTS ANALYSIS")
    print("=" * 80)

    answer = result.get('answer', '')
    sources = result.get('sources', [])
    all_retrieved = result.get('all_retrieved', len(sources))

    print(f"\n✅ Query Success: {result.get('success', False)}")
    print(f"\n📊 Retrieval Stats:")
    print(f"   - Documents retrieved: {all_retrieved}")
    print(f"   - Sources in response: {len(sources)}")

    # Citation analysis
    cited_numbers = get_cited_numbers(answer)

    print(f"\n🔗 Citation Analysis:")
    print(f"   - Citations found: {len(cited_numbers)}")
    print(f"   - Citation numbers: {sorted(cited_numbers)}")
    print(f"   - All sources cited: {len(cited_numbers) == len(sources)}")

    # Check citation consistency
    print(f"\n🔍 Citation Consistency Check:")

    if len(cited_numbers) >= 2:
        print(f"   ✅ PASS: Found {len(cited_numbers)} citations (≥2 required)")
    else:
        print(f"   ❌ FAIL: Only {len(cited_numbers)} citations (2 required)")

    # Check all cited numbers are valid
    invalid_citations = [n for n in cited_numbers if n > len(sources)]
    if not invalid_citations:
        print(f"   ✅ PASS: All citations reference valid sources")
    else:
        print(f"   ❌ FAIL: Invalid citations: {invalid_citations}")

    # Check answer quality
    print(f"\n📝 Answer Quality:")
    print(f"   - Answer length: {len(answer)} characters")
    print(f"   - Contains 'transformer': {'transformer' in answer.lower()}")
    print(f"   - Contains 'attention': {'attention' in answer.lower()}")

    # Display answer preview
    print(f"\n💬 Answer Preview (first 500 chars):")
    print("-" * 80)
    print(answer[:500] + "..." if len(answer) > 500 else answer)
    print("-" * 80)

    # Display sources
    print(f"\n📚 Sources ({len(sources)} total):")
    for i, source in enumerate(sources, 1):
        title = source.get('title', source.get('metadata', {}).get('title', 'Unknown'))
        score = source.get('score', 0)
        index = source.get('index', i)
        print(f"   [{index}] {title[:70]}... (score: {score:.4f})")

    # Performance metrics
    if result.get('metadata'):
        metadata = result['metadata']
        print(f"\n⏱️  Performance:")
        print(f"   - Retrieval:  {metadata['retrieval_time']:.3f}s")
        print(f"   - Ranking:    {metadata['ranking_time']:.3f}s")
        print(f"   - Generation: {metadata['generation_time']:.3f}s")
        print(f"   - Total:      {metadata['total_time']:.3f}s")

    # Citation stats
    stats = CitationFormatter.get_citation_stats(answer, sources)
    print(f"\n📈 Citation Statistics:")
    print(f"   - Total sources: {stats['total_sources']}")
    print(f"   - Cited sources: {stats['cited_sources']}")
    print(f"   - Uncited sources: {stats['uncited_sources']}")
    print(f"   - Total citations: {stats['total_citations']}")
    print(f"   - Unique citations: {stats['unique_citations']}")
    print(f"   - Citation rate: {stats['citation_rate']:.1%}")

    # Save detailed results
    output_file = "citation_test_results.json"
    with open(output_file, 'w') as f:
        # Prepare serializable result
        serializable_result = {
            'query': test_query,
            'answer': answer,
            'sources': sources,
            'citation_stats': stats,
            'metadata': result.get('metadata', {}),
            'cited_numbers': sorted(cited_numbers)
        }
        json.dump(serializable_result, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {output_file}")

    # Final verdict
    print("\n" + "=" * 80)
    print("  FINAL VERDICT")
    print("=" * 80)

    passed_tests = 0
    total_tests = 4

    # Test 1: Query succeeded
    if result.get('success', False):
        print("✅ Test 1: Query succeeded")
        passed_tests += 1
    else:
        print("❌ Test 1: Query failed")

    # Test 2: At least 2 citations
    if len(cited_numbers) >= 2:
        print(f"✅ Test 2: Found {len(cited_numbers)} citations (≥2)")
        passed_tests += 1
    else:
        print(f"❌ Test 2: Only {len(cited_numbers)} citations (need ≥2)")

    # Test 3: All citations are valid
    if not invalid_citations:
        print("✅ Test 3: All citations are valid")
        passed_tests += 1
    else:
        print(f"❌ Test 3: Invalid citations found: {invalid_citations}")

    # Test 4: Answer contains relevant content
    has_relevant_content = (
        'transformer' in answer.lower() or
        'attention' in answer.lower()
    )
    if has_relevant_content:
        print("✅ Test 4: Answer contains relevant content")
        passed_tests += 1
    else:
        print("❌ Test 4: Answer lacks relevant content")

    print(f"\n🎯 Score: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print(f"⚠️  {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = test_transformer_query()
    sys.exit(exit_code)

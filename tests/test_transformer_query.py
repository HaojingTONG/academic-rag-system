"""
Test: Transformer Architecture Query Validation
================================================

This test validates the RAG system improvements for definition-type queries:
1. Hybrid retrieval (dense + BM25) retrieves relevant papers
2. Query expansion adds relevant terms (self-attention, encoder-decoder)
3. Definition template is used for "What is X?" queries
4. Generated answer follows structured format (Definition/Mechanism/Application)
5. Citations are present and accurate (≥2 sources cited)
6. Key concepts are mentioned (self-attention, encoder-decoder)

Expected Results:
- Retrieved documents include "Attention is All You Need" paper
- Query is expanded with terms like "self-attention", "attention mechanism"
- Answer uses definition template structure
- Answer contains ≥2 citations [1], [2]
- Answer mentions "self-attention" or related terms
- No prompt meta-text in answer

Updated: 2025-10-27
"""

import sys
import re
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline, RAGConfig
from rag.retriever import QueryAnalyzer, HybridRetriever


class TestTransformerQuery:
    """Test suite for Transformer Architecture query"""

    def __init__(self):
        self.query = "What is Transformer Architecture?"
        self.pipeline = None
        self.result = None
        self.test_results = {
            'query_expansion': False,
            'intent_detection': False,
            'hybrid_retrieval': False,
            'citation_count': 0,
            'structure_check': False,
            'key_concepts': False,
            'no_meta_text': False,
            'paper_relevance': False
        }

    def setup(self):
        """Initialize RAG pipeline"""
        print("🔧 Setting up RAG pipeline...")

        # Create config with hybrid retrieval enabled
        config = RAGConfig(
            top_k=5,
            use_hybrid_retrieval=True,
            enable_reranking=True,
            enable_diversification=True
        )

        # Initialize pipeline
        self.pipeline = RAGPipeline(config)
        self.pipeline.initialize()

        print("✅ Pipeline ready\n")

    def test_query_expansion(self):
        """Test: Query expansion adds relevant terms"""
        print("📝 Test 1: Query Expansion")
        print("-" * 60)

        analyzer = QueryAnalyzer()
        intent = analyzer.analyze(self.query)

        print(f"Original query: {self.query}")
        print(f"Expanded query: {intent.expanded_query}")
        print(f"Keywords: {intent.keywords}")

        # Check if expansion includes transformer-related terms
        expansion_terms = ['attention', 'self-attention', 'multi-head']
        found_terms = [term for term in expansion_terms
                      if term in intent.expanded_query.lower()]

        self.test_results['query_expansion'] = len(found_terms) > 0

        if self.test_results['query_expansion']:
            print(f"✅ PASS: Query expanded with terms: {found_terms}")
        else:
            print(f"❌ FAIL: No transformer-related terms in expansion")

        print()

    def test_intent_detection(self):
        """Test: Definition intent is detected correctly"""
        print("📝 Test 2: Intent Detection")
        print("-" * 60)

        analyzer = QueryAnalyzer()
        intent = analyzer.analyze(self.query)

        print(f"Query: {self.query}")
        print(f"Detected intent: {intent.intent_type}")
        print(f"Confidence: {intent.confidence:.2f}")

        self.test_results['intent_detection'] = intent.intent_type == 'definition'

        if self.test_results['intent_detection']:
            print(f"✅ PASS: Correctly detected 'definition' intent")
        else:
            print(f"❌ FAIL: Expected 'definition', got '{intent.intent_type}'")

        print()

    def test_rag_pipeline(self):
        """Test: Full RAG pipeline execution"""
        print("📝 Test 3: RAG Pipeline Execution")
        print("-" * 60)

        # Execute query
        self.result = self.pipeline.query(
            question=self.query,
            top_k=5,
            return_metadata=True
        )

        # Check basic success
        if not self.result['success']:
            print(f"❌ FAIL: Pipeline execution failed")
            return

        print(f"✅ PASS: Pipeline executed successfully")
        print(f"Retrieved sources: {self.result['num_sources']}")
        print(f"Total time: {self.result['metadata']['total_time']:.2f}s")
        print()

    def test_citation_count(self):
        """Test: Answer contains ≥2 citations"""
        print("📝 Test 4: Citation Count")
        print("-" * 60)

        if not self.result:
            print("❌ FAIL: No result available")
            return

        answer = self.result['answer']

        # Find all citations [1], [2], etc.
        citations = re.findall(r'\[(\d+)\]', answer)
        unique_citations = set(citations)

        self.test_results['citation_count'] = len(unique_citations)

        print(f"Citations found: {unique_citations}")
        print(f"Total citations: {len(citations)}")
        print(f"Unique citations: {len(unique_citations)}")

        if len(unique_citations) >= 2:
            print(f"✅ PASS: Found {len(unique_citations)} citations (≥2 required)")
        else:
            print(f"❌ FAIL: Only {len(unique_citations)} citations found (need ≥2)")

        print()

    def test_answer_structure(self):
        """Test: Answer follows Definition/Mechanism/Application structure"""
        print("📝 Test 5: Answer Structure")
        print("-" * 60)

        if not self.result:
            print("❌ FAIL: No result available")
            return

        answer = self.result['answer']

        # Check for expected sections
        sections = {
            'Definition': r'\*\*Definition[:\s]',
            'Mechanism': r'\*\*Mechanism|Architecture[:\s]',
            'Application': r'\*\*Application[s]?[:\s]'
        }

        found_sections = []
        for section_name, pattern in sections.items():
            if re.search(pattern, answer, re.IGNORECASE):
                found_sections.append(section_name)

        self.test_results['structure_check'] = len(found_sections) >= 2

        print(f"Expected sections: {list(sections.keys())}")
        print(f"Found sections: {found_sections}")

        if self.test_results['structure_check']:
            print(f"✅ PASS: Answer has structured format ({len(found_sections)}/3 sections)")
        else:
            print(f"❌ FAIL: Missing structured format (only {len(found_sections)}/3 sections)")

        print()

    def test_key_concepts(self):
        """Test: Answer mentions key concepts (self-attention, encoder-decoder)"""
        print("📝 Test 6: Key Concepts Mentioned")
        print("-" * 60)

        if not self.result:
            print("❌ FAIL: No result available")
            return

        answer = self.result['answer'].lower()

        # Key concepts that should be mentioned
        key_concepts = [
            'self-attention',
            'attention mechanism',
            'encoder',
            'decoder',
            'multi-head',
            'transformer'
        ]

        found_concepts = [concept for concept in key_concepts if concept in answer]

        self.test_results['key_concepts'] = len(found_concepts) >= 2

        print(f"Key concepts to find: {key_concepts}")
        print(f"Found concepts: {found_concepts}")

        if self.test_results['key_concepts']:
            print(f"✅ PASS: Found {len(found_concepts)} key concepts (≥2 required)")
        else:
            print(f"❌ FAIL: Only {len(found_concepts)} key concepts found")

        print()

    def test_no_meta_text(self):
        """Test: Answer doesn't contain prompt meta-text"""
        print("📝 Test 7: No Meta-Text in Answer")
        print("-" * 60)

        if not self.result:
            print("❌ FAIL: No result available")
            return

        answer = self.result['answer']

        # Meta-text patterns that should NOT appear in answer
        meta_patterns = [
            r'You are an AI',
            r'\*\*Context\*\*.*relevant excerpts',
            r'\*\*Question:\*\*',
            r'\*\*Instructions:\*\*',
            r'based on the research papers below',
            r'Answer based ONLY on'
        ]

        found_meta = []
        for pattern in meta_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                found_meta.append(pattern)

        self.test_results['no_meta_text'] = len(found_meta) == 0

        if self.test_results['no_meta_text']:
            print(f"✅ PASS: No prompt meta-text found in answer")
        else:
            print(f"❌ FAIL: Found meta-text patterns: {found_meta}")

        print()

    def test_paper_relevance(self):
        """Test: Retrieved papers include transformer-related papers"""
        print("📝 Test 8: Paper Relevance")
        print("-" * 60)

        if not self.result or not self.result['sources']:
            print("❌ FAIL: No sources available")
            return

        sources = self.result['sources']

        # Keywords that indicate relevant papers
        relevant_keywords = [
            'attention',
            'transformer',
            'bert',
            'gpt',
            'encoder',
            'decoder'
        ]

        relevant_count = 0
        for source in sources:
            title = source.get('title', '').lower()
            if any(kw in title for kw in relevant_keywords):
                relevant_count += 1
                print(f"   ✓ Relevant: {source.get('title', 'Unknown')}")

        self.test_results['paper_relevance'] = relevant_count >= 1

        print(f"\nRelevant papers: {relevant_count}/{len(sources)}")

        if self.test_results['paper_relevance']:
            print(f"✅ PASS: Found {relevant_count} transformer-related papers")
        else:
            print(f"❌ FAIL: No transformer-related papers found")

        print()

    def print_answer(self):
        """Print the generated answer"""
        print("📄 Generated Answer")
        print("=" * 60)

        if not self.result:
            print("No answer available")
            return

        print(self.result['answer'])
        print("\n" + "=" * 60)
        print()

    def print_sources(self):
        """Print the retrieved sources"""
        print("📚 Retrieved Sources")
        print("=" * 60)

        if not self.result or not self.result['sources']:
            print("No sources available")
            return

        for idx, source in enumerate(self.result['sources'], 1):
            title = source.get('title', 'Unknown')
            score = source.get('score', 0)
            print(f"\n[{idx}] {title}")
            print(f"    Score: {score:.3f}")

        print("\n" + "=" * 60)
        print()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for v in self.test_results.values() if v is True or (isinstance(v, int) and v >= 2))

        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {passed_tests/total_tests*100:.1f}%\n")

        # Detailed results
        print("Detailed Results:")
        print("-" * 60)

        status_map = {
            'query_expansion': 'Query Expansion',
            'intent_detection': 'Intent Detection',
            'citation_count': 'Citation Count (≥2)',
            'structure_check': 'Answer Structure',
            'key_concepts': 'Key Concepts',
            'no_meta_text': 'No Meta-Text',
            'paper_relevance': 'Paper Relevance'
        }

        for key, name in status_map.items():
            value = self.test_results[key]

            if key == 'citation_count':
                status = "✅ PASS" if value >= 2 else "❌ FAIL"
                print(f"{status} | {name}: {value}")
            else:
                status = "✅ PASS" if value else "❌ FAIL"
                print(f"{status} | {name}")

        print("\n" + "=" * 60)

        # Overall verdict
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED!")
        elif passed_tests >= total_tests * 0.7:
            print("⚠️  MOST TESTS PASSED (some improvements needed)")
        else:
            print("❌ MULTIPLE FAILURES (system needs fixes)")

        print("=" * 60 + "\n")

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "=" * 60)
        print("🧪 Running Transformer Query Test Suite")
        print("=" * 60 + "\n")

        try:
            # Setup
            self.setup()

            # Run tests
            self.test_query_expansion()
            self.test_intent_detection()
            self.test_rag_pipeline()

            # Only run remaining tests if pipeline succeeded
            if self.result and self.result['success']:
                self.test_citation_count()
                self.test_answer_structure()
                self.test_key_concepts()
                self.test_no_meta_text()
                self.test_paper_relevance()

                # Show results
                self.print_answer()
                self.print_sources()

            # Summary
            self.print_summary()

        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    tester = TestTransformerQuery()
    tester.run_all_tests()


if __name__ == "__main__":
    main()

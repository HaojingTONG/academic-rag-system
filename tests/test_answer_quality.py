"""
Test: Answer Quality and Faithfulness Metrics
==============================================

This test validates answer quality enhancements:
1. Answers synthesize rather than copy
2. Faithfulness metrics work correctly
3. Post-generation summary injection works
4. Citation coverage meets thresholds
5. Verbatim copying is detected and minimized
6. Claim support is verified

Expected Results:
- Faithfulness score ≥ 0.7
- Citation coverage ≥ 50%
- Verbatim copying < 30%
- Answer includes summary paragraph
- No verbatim sentences from context
- Technical clarity maintained

Updated: 2025-10-27
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.answer_quality import (
    FaithfulnessChecker,
    AnswerEnhancer,
    QualityAnalyzer,
    check_answer_quality
)


class TestAnswerQuality:
    """Test suite for answer quality enhancements"""

    def __init__(self):
        self.test_results = {
            'faithfulness_check': False,
            'citation_coverage': False,
            'verbatim_detection': False,
            'summary_injection': False,
            'quality_metrics': False,
            'comprehensive_check': False
        }

    def test_faithfulness_checker(self):
        """Test: Faithfulness checker detects issues"""
        print("📝 Test 1: Faithfulness Checker")
        print("-" * 60)

        # Sample context and answers
        context = """
        The Transformer architecture uses self-attention mechanisms to process sequences in parallel.
        It consists of encoder and decoder stacks, each containing multiple layers.
        Multi-head attention allows the model to attend to different representation subspaces.
        Position embeddings provide sequence order information without recurrence.
        """

        # Good answer: cited, rephrased
        good_answer = """
        The Transformer is a neural network design that employs attention-based processing [1].
        Unlike recurrent models, it handles entire sequences simultaneously [2].
        Its architecture features stacked encoding and decoding components [1].
        Through multiple attention heads, it can focus on various aspects of the input [3].
        """

        # Bad answer: verbatim copying, no citations
        bad_answer = """
        The Transformer architecture uses self-attention mechanisms to process sequences in parallel.
        It consists of encoder and decoder stacks, each containing multiple layers.
        Multi-head attention allows the model to attend to different representation subspaces.
        """

        checker = FaithfulnessChecker()

        # Check good answer
        good_score = checker.check_faithfulness(good_answer, context)
        print(f"\nGood Answer Metrics:")
        print(f"  Overall Score: {good_score.score:.2f}")
        print(f"  Citation Coverage: {good_score.citation_coverage:.1%}")
        print(f"  Verbatim Copying: {good_score.verbatim_copying:.1%}")
        print(f"  Claim Support: {good_score.claim_support:.1%}")
        print(f"  Issues: {len(good_score.issues)}")

        # Check bad answer
        bad_score = checker.check_faithfulness(bad_answer, context)
        print(f"\nBad Answer Metrics:")
        print(f"  Overall Score: {bad_score.score:.2f}")
        print(f"  Citation Coverage: {bad_score.citation_coverage:.1%}")
        print(f"  Verbatim Copying: {bad_score.verbatim_copying:.1%}")
        print(f"  Claim Support: {bad_score.claim_support:.1%}")
        print(f"  Issues: {bad_score.issues}")

        # Verify good answer passes
        good_passes = (
            good_score.score >= 0.7 and
            good_score.citation_coverage >= 0.5 and
            good_score.verbatim_copying < 0.3
        )

        # Verify bad answer fails
        bad_fails = (
            bad_score.score < 0.7 or
            bad_score.citation_coverage < 0.5 or
            bad_score.verbatim_copying > 0.3
        )

        self.test_results['faithfulness_check'] = good_passes and bad_fails

        if self.test_results['faithfulness_check']:
            print(f"\n✅ PASS: Faithfulness checker works correctly")
        else:
            print(f"\n❌ FAIL: Faithfulness checker issues")

        print()

    def test_citation_coverage(self):
        """Test: Citation coverage calculation"""
        print("📝 Test 2: Citation Coverage")
        print("-" * 60)

        checker = FaithfulnessChecker()

        # Test various citation patterns
        test_cases = [
            ("This is a sentence [1]. Another sentence [2].", 1.0),
            ("Cited sentence [1]. Uncited sentence.", 0.5),
            ("No citations at all in this answer.", 0.0),
            ("Every. Single. Sentence [1]. Has [2]. Citations [3].", 1.0)
        ]

        all_passed = True

        for answer, expected_coverage in test_cases:
            coverage = checker._check_citation_coverage(answer)
            passed = abs(coverage - expected_coverage) < 0.01

            status = "✓" if passed else "✗"
            print(f"  {status} '{answer[:40]}...' → {coverage:.1%} (expected {expected_coverage:.1%})")

            if not passed:
                all_passed = False

        self.test_results['citation_coverage'] = all_passed

        if all_passed:
            print(f"\n✅ PASS: Citation coverage calculation correct")
        else:
            print(f"\n❌ FAIL: Citation coverage calculation issues")

        print()

    def test_verbatim_detection(self):
        """Test: Verbatim copying detection"""
        print("📝 Test 3: Verbatim Copying Detection")
        print("-" * 60)

        context = "The quick brown fox jumps over the lazy dog every single day."

        test_cases = [
            # (answer, should_detect_copying)
            ("The quick brown fox jumps over", True),  # Verbatim
            ("A fast brown fox leaps across", False),  # Rephrased
            ("Foxes are quick and brown", False),  # Different structure
            ("Every single day the quick brown fox jumps", True)  # Partial verbatim
        ]

        checker = FaithfulnessChecker(verbatim_threshold=5)
        all_correct = True

        for answer, should_detect in test_cases:
            copying_ratio = checker._check_verbatim_copying(answer, context)
            detected = copying_ratio > 0.2  # Threshold for detection

            correct = detected == should_detect
            status = "✓" if correct else "✗"

            print(f"  {status} '{answer}' → {copying_ratio:.1%} copying "
                  f"({'detected' if detected else 'not detected'})")

            if not correct:
                all_correct = False

        self.test_results['verbatim_detection'] = all_correct

        if all_correct:
            print(f"\n✅ PASS: Verbatim detection works correctly")
        else:
            print(f"\n❌ FAIL: Verbatim detection issues")

        print()

    def test_summary_injection(self):
        """Test: Summary paragraph injection"""
        print("📝 Test 4: Summary Injection")
        print("-" * 60)

        enhancer = AnswerEnhancer()

        # Answer without summary
        answer_no_summary = """
        The Transformer architecture is a neural network model [1].
        It uses self-attention mechanisms for processing [2].
        Applications include machine translation and text generation [3].
        """

        # Answer with summary
        answer_with_summary = """
        The Transformer architecture is a neural network model [1].

        **Summary:**
        This concept is well-established in the field.
        """

        sources = [
            {'title': 'Attention Is All You Need', 'score': 0.9},
            {'title': 'BERT Paper', 'score': 0.8}
        ]

        # Test 1: Should add summary
        enhanced_1 = enhancer.add_summary(answer_no_summary, "What is Transformer?", sources)
        has_summary_1 = '**Summary:**' in enhanced_1 or 'summary' in enhanced_1.lower()

        print(f"Test 1: Add summary to answer without summary")
        print(f"  Original length: {len(answer_no_summary)} chars")
        print(f"  Enhanced length: {len(enhanced_1)} chars")
        print(f"  Has summary: {has_summary_1}")
        print(f"  Status: {'✓' if has_summary_1 else '✗'}")

        # Test 2: Should NOT duplicate summary
        enhanced_2 = enhancer.add_summary(answer_with_summary, "What is Transformer?", sources)
        summary_count = enhanced_2.lower().count('summary')

        print(f"\nTest 2: Don't duplicate existing summary")
        print(f"  Summary count: {summary_count}")
        print(f"  Status: {'✓' if summary_count == 1 else '✗'}")

        self.test_results['summary_injection'] = has_summary_1 and summary_count == 1

        if self.test_results['summary_injection']:
            print(f"\n✅ PASS: Summary injection works correctly")
        else:
            print(f"\n❌ FAIL: Summary injection issues")

        print()

    def test_quality_metrics(self):
        """Test: Quality metrics calculation"""
        print("📝 Test 5: Quality Metrics")
        print("-" * 60)

        analyzer = QualityAnalyzer()

        sample_answer = """
        The Transformer architecture revolutionized Natural Language Processing [1].
        It employs Multi-Head Attention mechanisms for parallel sequence processing [2].
        Applications span Machine Translation, Text Summarization, and Question Answering [3].

        **Summary:**
        Transformers represent a paradigm shift in sequence modeling.
        """

        metrics = analyzer.analyze(sample_answer)

        print(f"Answer Metrics:")
        print(f"  Word count: {metrics.word_count}")
        print(f"  Sentence count: {metrics.sentence_count}")
        print(f"  Citation count: {metrics.citation_count}")
        print(f"  Unique citations: {metrics.unique_citations}")
        print(f"  Has summary: {metrics.has_summary}")
        print(f"  Technical terms: {metrics.technical_terms}")
        print(f"  Readability score: {metrics.readability_score:.2f}")

        # Verify metrics
        checks = [
            metrics.word_count > 0,
            metrics.sentence_count > 0,
            metrics.citation_count >= 3,
            metrics.unique_citations == 3,
            metrics.has_summary == True,
            metrics.technical_terms >= 3,  # Transformer, NLP, etc.
            0 <= metrics.readability_score <= 1
        ]

        self.test_results['quality_metrics'] = all(checks)

        if self.test_results['quality_metrics']:
            print(f"\n✅ PASS: Quality metrics calculated correctly")
        else:
            print(f"\n❌ FAIL: Quality metrics issues")
            print(f"  Failed checks: {[i for i, c in enumerate(checks) if not c]}")

        print()

    def test_comprehensive_check(self):
        """Test: Comprehensive quality check"""
        print("📝 Test 6: Comprehensive Quality Check")
        print("-" * 60)

        context = """
        Transformers use self-attention to weight input tokens based on relevance.
        The architecture eliminates recurrence, enabling parallel processing.
        Position encodings inject sequence order information.
        Multi-head attention captures different representation aspects.
        """

        answer = """
        The Transformer model employs attention-based weighting of input elements [1].
        By removing sequential dependencies, it achieves efficient parallelization [2].
        Sequence positions are encoded explicitly through dedicated embeddings [3].
        Multiple attention mechanisms capture diverse contextual relationships [4].

        **Summary:**
        Transformers represent a fundamental shift from sequential to parallel architectures.
        """

        question = "How do Transformers work?"
        sources = [
            {'title': 'Attention Is All You Need', 'score': 0.95},
            {'title': 'BERT', 'score': 0.85}
        ]

        result = check_answer_quality(answer, context, question, sources)

        print(f"Comprehensive Check Results:")
        print(f"\nFaithfulness:")
        print(f"  Overall score: {result['faithfulness']['score']:.2f}")
        print(f"  Citation coverage: {result['faithfulness']['citation_coverage']:.1%}")
        print(f"  Claim support: {result['faithfulness']['claim_support']:.1%}")
        print(f"  Verbatim copying: {result['faithfulness']['verbatim_copying']:.1%}")
        print(f"  Issues: {len(result['faithfulness']['issues'])}")

        print(f"\nQuality:")
        print(f"  Word count: {result['quality']['word_count']}")
        print(f"  Citations: {result['quality']['citation_count']}")
        print(f"  Has summary: {result['quality']['has_summary']}")
        print(f"  Readability: {result['quality']['readability_score']:.2f}")

        print(f"\nOverall Score: {result['overall_score']:.2f}")

        # Verify thresholds
        passes = (
            result['faithfulness']['score'] >= 0.7 and
            result['faithfulness']['citation_coverage'] >= 0.5 and
            result['faithfulness']['verbatim_copying'] < 0.3 and
            result['quality']['has_summary'] == True and
            result['overall_score'] >= 0.7
        )

        self.test_results['comprehensive_check'] = passes

        if passes:
            print(f"\n✅ PASS: Comprehensive check passed all thresholds")
        else:
            print(f"\n❌ FAIL: Comprehensive check failed")

        print()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 Test Summary")
        print("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for v in self.test_results.values() if v)

        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {passed_tests/total_tests*100:.1f}%\n")

        # Detailed results
        print("Detailed Results:")
        print("-" * 60)

        status_map = {
            'faithfulness_check': 'Faithfulness Checker',
            'citation_coverage': 'Citation Coverage',
            'verbatim_detection': 'Verbatim Detection',
            'summary_injection': 'Summary Injection',
            'quality_metrics': 'Quality Metrics',
            'comprehensive_check': 'Comprehensive Check'
        }

        for key, name in status_map.items():
            value = self.test_results[key]
            status = "✅ PASS" if value else "❌ FAIL"
            print(f"{status} | {name}")

        print("\n" + "=" * 60)

        # Overall verdict
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED!")
        elif passed_tests >= total_tests * 0.8:
            print("✅ MOST TESTS PASSED (good quality)")
        elif passed_tests >= total_tests * 0.5:
            print("⚠️  HALF TESTS PASSED (needs improvement)")
        else:
            print("❌ MULTIPLE FAILURES (requires fixes)")

        print("=" * 60 + "\n")

    def run_all_tests(self):
        """Run all tests in sequence"""
        print("\n" + "=" * 60)
        print("🧪 Running Answer Quality Test Suite")
        print("=" * 60 + "\n")

        try:
            # Run tests
            self.test_faithfulness_checker()
            self.test_citation_coverage()
            self.test_verbatim_detection()
            self.test_summary_injection()
            self.test_quality_metrics()
            self.test_comprehensive_check()

            # Summary
            self.print_summary()

        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    tester = TestAnswerQuality()
    tester.run_all_tests()


if __name__ == "__main__":
    main()

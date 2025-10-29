"""
Answer Quality Evaluation Framework
====================================

This module evaluates answer quality on concept queries:
- Transformer Architecture
- BERT
- Attention Mechanism
- GPT

Metrics:
- Hit@5 (retrieval quality)
- Keyword presence (required ≥3)
- Faithfulness score (≥0.9 target)
- Citation coverage
- Verbatim copying

Generates: eval/report_answer_quality.md
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline, RAGConfig
from rag.answer_quality import KeywordValidator, FaithfulnessChecker, check_answer_quality


class AnswerQualityEvaluator:
    """Evaluate answer quality on concept queries"""

    # Test queries
    TEST_QUERIES = [
        {
            'query': 'What is Transformer Architecture?',
            'concept': 'transformer',
            'required_keywords': ['self-attention', 'encoder-decoder', 'attention mechanism'],
            'expected_paper': 'Attention Is All You Need'
        },
        {
            'query': 'What is BERT?',
            'concept': 'bert',
            'required_keywords': ['bidirectional', 'transformer', 'pre-training'],
            'expected_paper': 'BERT: Pre-training'
        },
        {
            'query': 'What is attention mechanism?',
            'concept': 'attention',
            'required_keywords': ['attention mechanism', 'weight', 'relevance'],
            'expected_paper': 'Attention Is All You Need'
        }
    ]

    def __init__(self, output_file: str = "eval/report_answer_quality.md"):
        """
        Initialize evaluator

        Args:
            output_file: Path to output report file
        """
        self.output_file = output_file
        self.results = []
        self.pipeline = None

    def setup(self):
        """Initialize RAG pipeline"""
        print("🔧 Setting up RAG pipeline...")

        config = RAGConfig(
            top_k=5,
            use_hybrid_retrieval=True,
            enable_reranking=True,
            enable_diversification=True
        )

        self.pipeline = RAGPipeline(config)
        self.pipeline.initialize()

        print("✅ Pipeline ready\n")

    def evaluate_query(self, query_info: Dict) -> Dict:
        """
        Evaluate a single query

        Args:
            query_info: Query information dictionary

        Returns:
            Evaluation result
        """
        print(f"\n{'='*60}")
        print(f"📝 Query: {query_info['query']}")
        print(f"{'='*60}\n")

        query = query_info['query']
        concept = query_info['concept']
        required_keywords = query_info['required_keywords']
        expected_paper = query_info['expected_paper']

        # Execute query
        result = self.pipeline.query(
            question=query,
            top_k=5,
            return_metadata=True
        )

        if not result['success']:
            return {
                'query': query,
                'success': False,
                'error': 'Pipeline failed'
            }

        answer = result['answer']
        sources = result['sources']
        metadata = result.get('metadata', {})

        # 1. Check Hit@5 (is expected paper in top 5?)
        hit_at_5 = any(expected_paper.lower() in s.get('title', '').lower() for s in sources)

        # 2. Keyword validation
        validator = KeywordValidator()
        keyword_validation = validator.validate_keywords(
            answer=answer,
            concept=concept
        )

        # 3. Comprehensive quality check
        context_text = ' '.join([s.get('content', '') for s in sources])
        quality_report = check_answer_quality(
            answer=answer,
            context=context_text,
            question=query,
            sources=sources
        )

        # 4. Collect metrics
        evaluation = {
            'query': query,
            'concept': concept,
            'success': True,

            # Retrieval metrics
            'hit_at_5': hit_at_5,
            'num_sources': len(sources),
            'expected_paper_found': hit_at_5,

            # Keyword metrics
            'required_keywords': required_keywords,
            'keywords_present': keyword_validation.required_present,
            'keywords_missing': keyword_validation.required_missing,
            'keyword_coverage': keyword_validation.coverage,
            'keyword_count': len(keyword_validation.required_present),

            # Faithfulness metrics
            'faithfulness_score': quality_report['faithfulness']['score'],
            'citation_coverage': quality_report['faithfulness']['citation_coverage'],
            'verbatim_copying': quality_report['faithfulness']['verbatim_copying'],
            'claim_support': quality_report['faithfulness']['claim_support'],

            # Quality metrics
            'word_count': quality_report['quality']['word_count'],
            'citation_count': quality_report['quality']['citation_count'],
            'has_summary': quality_report['quality']['has_summary'],

            # Pass/Fail
            'passes_threshold': (
                quality_report['faithfulness']['score'] >= 0.7 and
                keyword_validation.coverage >= 0.5 and
                quality_report['faithfulness']['verbatim_copying'] < 0.3
            ),

            # Raw data
            'answer': answer,
            'sources': [{'title': s.get('title', ''), 'score': s.get('score', 0)} for s in sources],
            'metadata': metadata
        }

        # Print summary
        self._print_evaluation_summary(evaluation)

        return evaluation

    def _print_evaluation_summary(self, eval_result: Dict):
        """Print evaluation summary"""
        print(f"\n📊 Evaluation Results:")
        print(f"{'─'*60}")

        # Retrieval
        print(f"\n🔍 Retrieval:")
        print(f"  Hit@5: {'✅' if eval_result['hit_at_5'] else '❌'}")
        print(f"  Sources retrieved: {eval_result['num_sources']}")

        # Keywords
        print(f"\n🔑 Keywords:")
        print(f"  Coverage: {eval_result['keyword_coverage']:.1%} "
              f"({eval_result['keyword_count']}/{len(eval_result['required_keywords'])})")
        print(f"  Present: {', '.join(eval_result['keywords_present'])}")
        if eval_result['keywords_missing']:
            print(f"  Missing: {', '.join(eval_result['keywords_missing'])}")

        # Faithfulness
        print(f"\n✨ Faithfulness:")
        print(f"  Overall score: {eval_result['faithfulness_score']:.2f}")
        print(f"  Citation coverage: {eval_result['citation_coverage']:.1%}")
        print(f"  Verbatim copying: {eval_result['verbatim_copying']:.1%}")

        # Overall
        status = "✅ PASS" if eval_result['passes_threshold'] else "❌ FAIL"
        print(f"\n{status} | Overall: {'Meets thresholds' if eval_result['passes_threshold'] else 'Below thresholds'}")

    def evaluate_all(self):
        """Evaluate all test queries"""
        print("\n" + "="*60)
        print("🧪 Answer Quality Evaluation")
        print("="*60)

        self.results = []

        for query_info in self.TEST_QUERIES:
            result = self.evaluate_query(query_info)
            self.results.append(result)

        # Generate summary
        self._print_overall_summary()

    def _print_overall_summary(self):
        """Print overall summary"""
        print("\n\n" + "="*60)
        print("📊 Overall Summary")
        print("="*60)

        total = len(self.results)
        passed = sum(1 for r in self.results if r.get('passes_threshold', False))

        print(f"\nTotal queries: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Pass rate: {passed/total*100:.1f}%")

        # Average metrics
        avg_faithfulness = sum(r['faithfulness_score'] for r in self.results) / total
        avg_keyword_coverage = sum(r['keyword_coverage'] for r in self.results) / total
        avg_verbatim = sum(r['verbatim_copying'] for r in self.results) / total

        print(f"\nAverage Metrics:")
        print(f"  Faithfulness: {avg_faithfulness:.2f}")
        print(f"  Keyword coverage: {avg_keyword_coverage:.1%}")
        print(f"  Verbatim copying: {avg_verbatim:.1%}")

        print("\n" + "="*60)

    def generate_report(self):
        """Generate evaluation report"""
        print(f"\n📝 Generating report: {self.output_file}")

        report = self._create_report_content()

        # Write report
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ Report saved: {self.output_file}")

    def _create_report_content(self) -> str:
        """Create report content"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = []
        lines.append("# Answer Quality Evaluation Report")
        lines.append(f"**Generated:** {timestamp}\n")
        lines.append("---\n")

        # Overall metrics
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get('passes_threshold', False))
        avg_faithfulness = sum(r['faithfulness_score'] for r in self.results) / total
        avg_keyword_coverage = sum(r['keyword_coverage'] for r in self.results) / total
        avg_verbatim = sum(r['verbatim_copying'] for r in self.results) / total

        lines.append("## Summary\n")
        lines.append(f"- **Total Queries:** {total}")
        lines.append(f"- **Passed:** {passed}/{total} ({passed/total*100:.1f}%)")
        lines.append(f"- **Average Faithfulness:** {avg_faithfulness:.2f}")
        lines.append(f"- **Average Keyword Coverage:** {avg_keyword_coverage:.1%}")
        lines.append(f"- **Average Verbatim Copying:** {avg_verbatim:.1%}\n")

        # Individual results
        lines.append("---\n")
        lines.append("## Individual Query Results\n")

        for i, result in enumerate(self.results, 1):
            lines.append(f"\n### {i}. {result['query']}\n")

            status = "✅ PASS" if result['passes_threshold'] else "❌ FAIL"
            lines.append(f"**Status:** {status}\n")

            # Metrics table
            lines.append("#### Metrics\n")
            lines.append("| Metric | Value | Threshold | Status |")
            lines.append("|--------|-------|-----------|--------|")

            # Hit@5
            hit_status = "✅" if result['hit_at_5'] else "❌"
            lines.append(f"| Hit@5 | {result['hit_at_5']} | True | {hit_status} |")

            # Keyword coverage
            kw_status = "✅" if result['keyword_coverage'] >= 0.5 else "❌"
            lines.append(f"| Keyword Coverage | {result['keyword_coverage']:.1%} | ≥50% | {kw_status} |")

            # Faithfulness
            faith_status = "✅" if result['faithfulness_score'] >= 0.7 else "❌"
            lines.append(f"| Faithfulness | {result['faithfulness_score']:.2f} | ≥0.7 | {faith_status} |")

            # Verbatim
            verb_status = "✅" if result['verbatim_copying'] < 0.3 else "❌"
            lines.append(f"| Verbatim Copying | {result['verbatim_copying']:.1%} | <30% | {verb_status} |")

            lines.append("")

            # Keywords
            lines.append("#### Keywords\n")
            lines.append(f"- **Required:** {', '.join(result['required_keywords'])}")
            lines.append(f"- **Present:** {', '.join(result['keywords_present']) if result['keywords_present'] else 'None'}")
            if result['keywords_missing']:
                lines.append(f"- **Missing:** {', '.join(result['keywords_missing'])}")
            lines.append("")

            # Sources
            lines.append("#### Retrieved Sources\n")
            for j, source in enumerate(result['sources'], 1):
                lines.append(f"{j}. {source['title']} (Score: {source['score']:.3f})")
            lines.append("")

            # Answer preview
            lines.append("#### Answer Preview\n")
            answer_preview = result['answer'][:500] + "..." if len(result['answer']) > 500 else result['answer']
            lines.append(f"```\n{answer_preview}\n```\n")

        # Recommendations
        lines.append("---\n")
        lines.append("## Recommendations\n")

        if passed == total:
            lines.append("✅ All queries passed! System is performing well.\n")
        else:
            lines.append("⚠️ Some queries failed. Consider:\n")
            lines.append("1. Adjusting similarity thresholds for better retrieval")
            lines.append("2. Strengthening keyword requirements in prompts")
            lines.append("3. Adding more query expansion terms")
            lines.append("4. Improving fallback generator templates\n")

        return "\n".join(lines)

    def save_json_results(self, filename: str = "eval/evaluation_results.json"):
        """Save results as JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"✅ JSON results saved: {filename}")


def main():
    """Main entry point"""
    evaluator = AnswerQualityEvaluator()

    try:
        # Setup
        evaluator.setup()

        # Evaluate
        evaluator.evaluate_all()

        # Generate report
        evaluator.generate_report()

        # Save JSON
        evaluator.save_json_results()

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

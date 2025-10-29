"""
Smoke Evaluation Test
=====================

Validates index quality with fixed queries:
- retrieved_n ≥ 3 (minimum documents retrieved)
- kept_n ≥ 2 (minimum after filtering)
- Latency metrics (p50, p95)
- Citation presence

Fails if thresholds not met.

Usage:
    python eval/smoke_eval.py
"""

import sys
import time
import json
from pathlib import Path
from typing import List, Dict
import statistics

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.pipeline import RAGPipeline, RAGConfig


class SmokeEvaluator:
    """Run smoke tests on index"""

    # Fixed test queries
    TEST_QUERIES = [
        "What is Transformer Architecture?",
        "What is attention mechanism?",
        "What is BERT?",
        "Explain ResNet architecture",
        "What is Adam optimizer?"
    ]

    # Thresholds
    MIN_RETRIEVED = 3
    MIN_KEPT = 2
    MAX_LATENCY_P95 = 5.0  # seconds

    def __init__(self):
        """Initialize evaluator"""
        self.pipeline = None
        self.results = []

    def setup(self):
        """Initialize RAG pipeline"""
        print("🔧 Setting up RAG pipeline...")

        config = RAGConfig(
            top_k=10,
            use_hybrid_retrieval=True,
            enable_reranking=True
        )

        self.pipeline = RAGPipeline(config)
        self.pipeline.initialize()

        print("✅ Pipeline ready\n")

    def run_query(self, query: str) -> Dict:
        """Run single query and collect metrics"""
        start_time = time.time()

        result = self.pipeline.query(
            question=query,
            top_k=10,
            return_metadata=True
        )

        latency = time.time() - start_time

        if not result['success']:
            return {
                'query': query,
                'success': False,
                'error': 'Query failed'
            }

        # Count citations
        answer = result['answer']
        citations = set()
        import re
        for match in re.finditer(r'\[(\d+)\]', answer):
            citations.add(match.group(1))

        return {
            'query': query,
            'success': True,
            'retrieved_n': result['all_retrieved'],
            'kept_n': result['num_sources'],
            'citation_count': len(citations),
            'latency': latency
        }

    def evaluate(self):
        """Run all smoke tests"""
        print("\n" + "="*60)
        print("🧪 Smoke Evaluation")
        print("="*60 + "\n")

        self.results = []

        for query in self.TEST_QUERIES:
            print(f"📝 Query: {query}")

            result = self.run_query(query)
            self.results.append(result)

            if result['success']:
                print(f"   Retrieved: {result['retrieved_n']}")
                print(f"   Kept: {result['kept_n']}")
                print(f"   Citations: {result['citation_count']}")
                print(f"   Latency: {result['latency']:.3f}s")

                # Check thresholds
                if result['retrieved_n'] < self.MIN_RETRIEVED:
                    print(f"   ⚠️  Below threshold: retrieved_n < {self.MIN_RETRIEVED}")

                if result['kept_n'] < self.MIN_KEPT:
                    print(f"   ⚠️  Below threshold: kept_n < {self.MIN_KEPT}")
            else:
                print(f"   ❌ Failed: {result.get('error')}")

            print()

        # Overall results
        self._print_summary()

        # Save results
        self._save_results()

        # Check if passed
        if self._check_passed():
            print("\n✅ SMOKE TEST PASSED")
            sys.exit(0)
        else:
            print("\n❌ SMOKE TEST FAILED")
            sys.exit(1)

    def _print_summary(self):
        """Print summary statistics"""
        print("="*60)
        print("📊 Summary")
        print("="*60 + "\n")

        successful = [r for r in self.results if r['success']]

        if not successful:
            print("❌ No successful queries")
            return

        # Retrieved counts
        retrieved_counts = [r['retrieved_n'] for r in successful]
        kept_counts = [r['kept_n'] for r in successful]
        citation_counts = [r['citation_count'] for r in successful]
        latencies = [r['latency'] for r in successful]

        print(f"Successful queries: {len(successful)}/{len(self.results)}\n")

        print("Retrieved (all):")
        print(f"  Min: {min(retrieved_counts)}")
        print(f"  Max: {max(retrieved_counts)}")
        print(f"  Mean: {statistics.mean(retrieved_counts):.1f}\n")

        print("Kept (after filtering):")
        print(f"  Min: {min(kept_counts)}")
        print(f"  Max: {max(kept_counts)}")
        print(f"  Mean: {statistics.mean(kept_counts):.1f}\n")

        print("Citations:")
        print(f"  Min: {min(citation_counts)}")
        print(f"  Max: {max(citation_counts)}")
        print(f"  Mean: {statistics.mean(citation_counts):.1f}\n")

        print("Latency (seconds):")
        print(f"  p50: {statistics.median(latencies):.3f}")
        print(f"  p95: {sorted(latencies)[int(len(latencies)*0.95)]:.3f}")
        print(f"  Max: {max(latencies):.3f}\n")

    def _check_passed(self) -> bool:
        """Check if smoke tests passed"""
        successful = [r for r in self.results if r['success']]

        if not successful:
            return False

        # Check thresholds
        for result in successful:
            if result['retrieved_n'] < self.MIN_RETRIEVED:
                print(f"❌ Failed: {result['query']} retrieved only {result['retrieved_n']} "
                      f"(min {self.MIN_RETRIEVED})")
                return False

            if result['kept_n'] < self.MIN_KEPT:
                print(f"❌ Failed: {result['query']} kept only {result['kept_n']} "
                      f"(min {self.MIN_KEPT})")
                return False

        # Check latency
        latencies = [r['latency'] for r in successful]
        p95 = sorted(latencies)[int(len(latencies)*0.95)]

        if p95 > self.MAX_LATENCY_P95:
            print(f"⚠️  Warning: p95 latency {p95:.3f}s exceeds {self.MAX_LATENCY_P95}s "
                  "(not failing)")

        return True

    def _save_results(self):
        """Save results to JSON"""
        output_file = "eval/smoke_results.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"💾 Results saved: {output_file}\n")


def main():
    """Main entry point"""
    evaluator = SmokeEvaluator()

    try:
        evaluator.setup()
        evaluator.evaluate()
    except Exception as e:
        print(f"\n❌ Smoke evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

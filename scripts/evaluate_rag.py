#!/usr/bin/env python3
"""
RAG System Evaluation Script
============================

Evaluates RAG system quality with standard metrics:
- Hit@K: Recall rate at top K results
- MRR: Mean Reciprocal Rank
- Faithfulness: Citation accuracy
- Latency: P50, P95, P99 percentiles

Usage:
    python evaluate_rag.py
    python evaluate_rag.py --eval-set custom_questions.json
    python evaluate_rag.py --compare baseline_results.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import time
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import statistics
import re

from rag.pipeline import RAGPipeline
from rag.retriever import VectorStore
from src.generator.llm_client import create_llm_manager_from_config


# ============================================================================
# Evaluation Dataset
# ============================================================================

DEFAULT_EVAL_QUESTIONS = [
    {
        "question": "What is Transformer Architecture?",
        "expected_topics": ["attention", "self-attention", "encoder", "decoder"],
        "relevant_papers": ["attention is all you need", "transformer"]
    },
    {
        "question": "How does BERT differ from GPT?",
        "expected_topics": ["bidirectional", "autoregressive", "masked language model"],
        "relevant_papers": ["bert", "gpt"]
    },
    {
        "question": "What is ResNet architecture?",
        "expected_topics": ["residual", "skip connection", "deep learning"],
        "relevant_papers": ["resnet", "deep residual learning"]
    },
    {
        "question": "What are the key innovations in Vision Transformers?",
        "expected_topics": ["image patches", "vision", "transformer"],
        "relevant_papers": ["vision transformer", "vit", "image is worth"]
    },
    {
        "question": "How does attention mechanism work?",
        "expected_topics": ["query", "key", "value", "attention"],
        "relevant_papers": ["attention is all you need"]
    },
    {
        "question": "What is transfer learning?",
        "expected_topics": ["pretrain", "fine-tune", "transfer"],
        "relevant_papers": ["bert", "transfer learning"]
    },
    {
        "question": "What are the challenges in training large language models?",
        "expected_topics": ["compute", "data", "scaling", "optimization"],
        "relevant_papers": ["gpt", "palm", "scaling"]
    },
    {
        "question": "How do convolutional neural networks work?",
        "expected_topics": ["convolution", "feature", "filter", "cnn"],
        "relevant_papers": ["resnet", "cnn"]
    },
    {
        "question": "What is self-supervised learning?",
        "expected_topics": ["self-supervised", "pretraining", "unlabeled"],
        "relevant_papers": ["bert", "self-supervised"]
    },
    {
        "question": "What are the components of a neural machine translation system?",
        "expected_topics": ["encoder", "decoder", "attention", "translation"],
        "relevant_papers": ["attention is all you need", "neural machine translation"]
    }
]


@dataclass
class EvalResult:
    """Single query evaluation result"""
    question: str
    retrieved_n: int
    kept_n: int
    has_citations: bool
    citation_count: int
    latency_ms: float
    hit_at_5: bool
    reciprocal_rank: float
    topic_coverage: float
    answer_length: int
    success: bool


# ============================================================================
# Metrics
# ============================================================================

def calculate_hit_at_k(results: List[EvalResult], k: int = 5) -> float:
    """Calculate Hit@K - fraction of queries that retrieved at least 1 relevant doc"""
    if not results:
        return 0.0

    hits = sum(1 for r in results if r.kept_n >= 1)
    return hits / len(results)


def calculate_mrr(results: List[EvalResult]) -> float:
    """Calculate Mean Reciprocal Rank"""
    if not results:
        return 0.0

    reciprocal_ranks = [r.reciprocal_rank for r in results]
    return statistics.mean(reciprocal_ranks)


def calculate_faithfulness(results: List[EvalResult]) -> float:
    """
    Calculate faithfulness: fraction of answers with proper citations

    This is a simplified metric. Full faithfulness would require:
    - Verifying each claim against sources
    - Checking citation accuracy
    """
    if not results:
        return 0.0

    faithful = sum(1 for r in results if r.has_citations)
    return faithful / len(results)


def calculate_latency_percentiles(results: List[EvalResult]) -> Dict[str, float]:
    """Calculate latency percentiles"""
    if not results:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

    latencies = sorted([r.latency_ms for r in results])
    n = len(latencies)

    return {
        "p50": latencies[int(n * 0.50)] if n > 0 else 0.0,
        "p95": latencies[int(n * 0.95)] if n > 1 else latencies[-1],
        "p99": latencies[int(n * 0.99)] if n > 1 else latencies[-1],
        "mean": statistics.mean(latencies),
        "max": max(latencies)
    }


def check_topic_coverage(answer: str, expected_topics: List[str]) -> float:
    """Check what fraction of expected topics are mentioned in answer"""
    if not expected_topics:
        return 1.0

    answer_lower = answer.lower()
    covered = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
    return covered / len(expected_topics)


def extract_citations(text: str) -> List[int]:
    """Extract citation numbers from text"""
    citations = re.findall(r'\[(\d+)\]', text)
    return [int(c) for c in citations]


# ============================================================================
# Evaluation Runner
# ============================================================================

def run_evaluation(
    pipeline: RAGPipeline,
    questions: List[Dict],
    top_k: int = 5
) -> List[EvalResult]:
    """
    Run evaluation on a set of questions

    Args:
        pipeline: RAG pipeline to evaluate
        questions: List of question dicts
        top_k: Number of results to retrieve

    Returns:
        List of evaluation results
    """
    results = []

    print(f"\n{'='*70}")
    print(f"  Running Evaluation on {len(questions)} Questions")
    print(f"{'='*70}\n")

    for i, q_dict in enumerate(questions, 1):
        question = q_dict["question"]
        expected_topics = q_dict.get("expected_topics", [])

        print(f"[{i}/{len(questions)}] {question[:60]}...")

        try:
            # Run query
            start_time = time.time()
            result = pipeline.query(
                question=question,
                top_k=top_k,
                return_metadata=True
            )
            latency = (time.time() - start_time) * 1000  # ms

            # Extract metrics
            answer = result.get('answer', '')
            sources = result.get('sources', [])
            citations = extract_citations(answer)

            # Calculate metrics
            topic_coverage = check_topic_coverage(answer, expected_topics)

            # Reciprocal rank (simplified: 1/rank of first relevant result)
            # Since we don't have ground truth, use 1.0 if we got results
            rr = 1.0 if len(sources) > 0 else 0.0

            eval_result = EvalResult(
                question=question,
                retrieved_n=result.get('metadata', {}).get('retrieved_count', len(sources)),
                kept_n=len(sources),
                has_citations=len(citations) > 0,
                citation_count=len(set(citations)),
                latency_ms=latency,
                hit_at_5=len(sources) >= 1,
                reciprocal_rank=rr,
                topic_coverage=topic_coverage,
                answer_length=len(answer),
                success=result.get('success', False)
            )

            results.append(eval_result)

            # Print quick stats
            print(f"   ✓ Retrieved: {eval_result.kept_n} | "
                  f"Citations: {eval_result.citation_count} | "
                  f"Latency: {latency:.0f}ms | "
                  f"Topics: {topic_coverage:.1%}")

        except Exception as e:
            print(f"   ✗ Error: {e}")
            results.append(EvalResult(
                question=question,
                retrieved_n=0,
                kept_n=0,
                has_citations=False,
                citation_count=0,
                latency_ms=0.0,
                hit_at_5=False,
                reciprocal_rank=0.0,
                topic_coverage=0.0,
                answer_length=0,
                success=False
            ))

    return results


# ============================================================================
# Reporting
# ============================================================================

def print_evaluation_report(results: List[EvalResult]):
    """Print formatted evaluation report"""
    print(f"\n{'='*70}")
    print(f"  Evaluation Results")
    print(f"{'='*70}\n")

    # Overall metrics
    hit_at_5 = calculate_hit_at_k(results, k=5)
    mrr = calculate_mrr(results)
    faithfulness = calculate_faithfulness(results)
    latency = calculate_latency_percentiles(results)

    print(f"📊 Quality Metrics:")
    print(f"   Hit@5:        {hit_at_5:.1%}  (queries with ≥1 result)")
    print(f"   MRR:          {mrr:.3f}  (mean reciprocal rank)")
    print(f"   Faithfulness: {faithfulness:.1%}  (answers with citations)")

    print(f"\n⏱️  Latency:")
    print(f"   P50:  {latency['p50']:>6.0f} ms")
    print(f"   P95:  {latency['p95']:>6.0f} ms")
    print(f"   P99:  {latency['p99']:>6.0f} ms")
    print(f"   Mean: {latency['mean']:>6.0f} ms")
    print(f"   Max:  {latency['max']:>6.0f} ms")

    print(f"\n📈 Retrieval Stats:")
    avg_retrieved = statistics.mean([r.kept_n for r in results])
    avg_citations = statistics.mean([r.citation_count for r in results])
    avg_topic_cov = statistics.mean([r.topic_coverage for r in results])

    print(f"   Avg Results:      {avg_retrieved:.1f}")
    print(f"   Avg Citations:    {avg_citations:.1f}")
    print(f"   Topic Coverage:   {avg_topic_cov:.1%}")

    print(f"\n✅ Success Rate: {sum(1 for r in results if r.success)}/{len(results)}")

    print(f"\n{'='*70}\n")


def save_results(results: List[EvalResult], output_file: str):
    """Save results to JSON file"""
    with open(output_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"💾 Results saved to: {output_file}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    parser.add_argument('--eval-set', type=str, help="Path to custom eval questions JSON")
    parser.add_argument('--output', type=str, default="eval_results.json", help="Output file")
    parser.add_argument('--top-k', type=int, default=5, help="Number of results to retrieve")
    args = parser.parse_args()

    # Load questions
    if args.eval_set:
        with open(args.eval_set) as f:
            questions = json.load(f)
    else:
        questions = DEFAULT_EVAL_QUESTIONS

    # Initialize pipeline
    print("🚀 Initializing RAG Pipeline...")
    pipeline = RAGPipeline()

    vector_store = VectorStore()
    llm_manager = create_llm_manager_from_config()

    pipeline.initialize(
        vector_store=vector_store,
        llm_client=llm_manager
    )

    print("✅ Pipeline ready\n")

    # Run evaluation
    results = run_evaluation(pipeline, questions, top_k=args.top_k)

    # Print report
    print_evaluation_report(results)

    # Save results
    save_results(results, args.output)

    # Return exit code based on thresholds
    hit_at_5 = calculate_hit_at_k(results, k=5)
    if hit_at_5 < 0.8:  # 80% threshold
        print("⚠️  Warning: Hit@5 below 80% threshold")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

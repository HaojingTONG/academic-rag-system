"""
Relevance Filter - Smart Threshold-Based Filtering
==================================================

Filters retrieved documents based on relevance thresholds:
1. Score-based filtering (cosine similarity)
2. Query-document semantic match
3. Adaptive thresholds based on query type
"""

from typing import List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rag.retriever import RetrievalResult
except ImportError:
    from typing import Any as RetrievalResult


@dataclass
class RelevanceConfig:
    """Configuration for relevance filtering"""
    # Score thresholds
    min_vector_score: float = 0.35  # Minimum cosine similarity
    min_rerank_score: float = 0.30  # Minimum reranker score
    min_combined_score: float = 0.25  # Minimum hybrid score

    # Adaptive thresholds based on query type
    general_query_threshold: float = 0.20  # Lower threshold for general questions
    specific_query_threshold: float = 0.40  # Higher threshold for specific questions

    # Safety settings
    min_keep: int = 3  # Always keep at least this many results (if available)
    max_keep: int = 10  # Never keep more than this many


class RelevanceFilter:
    """
    Filter retrieved documents based on relevance scores

    This addresses the "irrelevant retrieval" problem by:
    1. Removing low-score results
    2. Adapting thresholds based on query type
    3. Ensuring minimum quality for answer generation
    """

    def __init__(self, config: Optional[RelevanceConfig] = None):
        """
        Initialize relevance filter

        Args:
            config: Filtering configuration
        """
        self.config = config or RelevanceConfig()

    def filter_results(
        self,
        results: List[RetrievalResult],
        query: str,
        is_general_query: bool = False
    ) -> List[RetrievalResult]:
        """
        Filter results based on relevance scores

        Args:
            results: Retrieved documents
            query: Original query (for adaptive thresholding)
            is_general_query: Whether this is a general knowledge query

        Returns:
            Filtered results
        """
        if not results:
            return []

        # Determine threshold based on query type
        if is_general_query:
            threshold = self.config.general_query_threshold
        else:
            threshold = self.config.specific_query_threshold

        # Filter by score
        filtered = []
        for result in results:
            # Get the best score from available scores
            best_score = self._get_best_score(result)

            if best_score >= threshold:
                filtered.append(result)

        # Ensure we keep minimum results if available
        if len(filtered) < self.config.min_keep and len(results) >= self.config.min_keep:
            # Keep top min_keep even if below threshold
            filtered = sorted(results, key=self._get_best_score, reverse=True)[:self.config.min_keep]

        # Cap at maximum
        if len(filtered) > self.config.max_keep:
            filtered = sorted(filtered, key=self._get_best_score, reverse=True)[:self.config.max_keep]

        return filtered

    def _get_best_score(self, result: RetrievalResult) -> float:
        """Get the best available score from a result"""
        scores = []

        if hasattr(result, 'rerank_score') and result.rerank_score is not None:
            scores.append(result.rerank_score)
        if hasattr(result, 'combined_score') and result.combined_score is not None:
            scores.append(result.combined_score)
        if hasattr(result, 'vector_score') and result.vector_score is not None:
            scores.append(result.vector_score)

        return max(scores) if scores else 0.0

    def check_retrieval_quality(
        self,
        results: List[RetrievalResult],
        is_general_query: bool = False
    ) -> dict:
        """
        Check if retrieval quality is acceptable

        Returns:
            Dict with quality assessment:
            - is_acceptable: bool
            - reason: str (if not acceptable)
            - avg_score: float
            - num_above_threshold: int
        """
        if not results:
            return {
                'is_acceptable': False,
                'reason': 'no_results',
                'avg_score': 0.0,
                'num_above_threshold': 0
            }

        # Calculate average score
        scores = [self._get_best_score(r) for r in results]
        avg_score = sum(scores) / len(scores)

        # Determine threshold
        threshold = (self.config.general_query_threshold if is_general_query
                    else self.config.specific_query_threshold)

        # Count results above threshold
        num_above_threshold = sum(1 for s in scores if s >= threshold)

        # Check if acceptable
        is_acceptable = (
            num_above_threshold >= self.config.min_keep and
            avg_score >= threshold
        )

        result = {
            'is_acceptable': is_acceptable,
            'avg_score': avg_score,
            'num_above_threshold': num_above_threshold,
            'total_results': len(results),
            'threshold': threshold
        }

        if not is_acceptable:
            if num_above_threshold == 0:
                result['reason'] = 'all_below_threshold'
            elif num_above_threshold < self.config.min_keep:
                result['reason'] = 'insufficient_quality_results'
            elif avg_score < threshold:
                result['reason'] = 'low_average_score'
            else:
                result['reason'] = 'unknown'

        return result


# Convenience function for pipeline integration
def filter_by_relevance(
    results: List[RetrievalResult],
    query: str,
    is_general_query: bool = False,
    config: Optional[RelevanceConfig] = None
) -> List[RetrievalResult]:
    """
    Filter results by relevance

    Args:
        results: Retrieved documents
        query: Original query
        is_general_query: Whether this is a general knowledge query
        config: Optional filtering configuration

    Returns:
        Filtered results
    """
    filter = RelevanceFilter(config)
    return filter.filter_results(results, query, is_general_query)


def check_retrieval_quality(
    results: List[RetrievalResult],
    is_general_query: bool = False,
    config: Optional[RelevanceConfig] = None
) -> dict:
    """
    Check retrieval quality

    Args:
        results: Retrieved documents
        is_general_query: Whether this is a general knowledge query
        config: Optional filtering configuration

    Returns:
        Quality assessment dict
    """
    filter = RelevanceFilter(config)
    return filter.check_retrieval_quality(results, is_general_query)

"""
Ranking and Filtering Module for Academic RAG System
====================================================

This module provides post-retrieval ranking and filtering:
- Cross-encoder reranking
- Maximal Marginal Relevance (MMR) for diversity
- Relevance filtering
- Result deduplication

Replaces ranking functionality from:
- src/retriever/advanced_retrieval.py (reranking part)
- src/retriever/enhanced_vector_retrieval.py (filtering part)
"""

import numpy as np
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config
try:
    from configs import config
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False

# Import RetrievalResult from retriever
try:
    from rag.retriever import RetrievalResult
except ImportError:
    # Fallback: define here
    @dataclass
    class RetrievalResult:
        doc_id: str
        content: str
        metadata: Dict
        vector_score: float = 0.0
        bm25_score: float = 0.0
        combined_score: float = 0.0
        rerank_score: float = 0.0


# ============================================================================
# Cross-Encoder Reranker
# ============================================================================

class CrossEncoderRanker:
    """
    Cross-encoder based reranking

    Supports multiple reranking models:
    - bge-reranker-large (default, highest quality)
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fallback)
    - Feature-based scoring (final fallback)
    """

    def __init__(self, use_model: bool = True, model_name: str = None):
        """
        Initialize reranker

        Args:
            use_model: Whether to use actual cross-encoder model (if False, uses features)
            model_name: Specific model to use (if None, tries bge-reranker-large then ms-marco)
        """
        self.use_model = use_model
        self.model = None
        self.model_type = "feature"  # feature, bge, cross-encoder

        # Feature weights for reranking
        self.feature_weights = {
            'query_overlap': 0.25,
            'section_relevance': 0.20,
            'content_quality': 0.20,
            'length_penalty': 0.10,
            'has_formulas': 0.10,
            'has_citations': 0.10,
            'position_penalty': 0.05  # Penalize later positions slightly
        }

        # Section relevance scores
        self.section_relevance = {
            'abstract': 0.9,
            'introduction': 0.75,
            'methodology': 1.0,
            'method': 1.0,
            'results': 0.85,
            'experiment': 0.85,
            'discussion': 0.70,
            'conclusion': 0.75,
            'related_work': 0.65,
            'content': 0.60
        }

        # Try to load reranker model if requested
        if self.use_model:
            # Try bge-reranker-large first (best quality)
            if model_name is None or model_name == "bge-reranker-large":
                try:
                    from FlagEmbedding import FlagReranker
                    self.model = FlagReranker('BAAI/bge-reranker-large', use_fp16=False)
                    self.model_type = "bge"
                    print(f"✅ Loaded reranker: BAAI/bge-reranker-large")
                except Exception as e:
                    print(f"⚠️ Could not load bge-reranker-large: {e}")
                    if model_name == "bge-reranker-large":
                        # If specifically requested, don't try fallback
                        self.use_model = False

            # Fallback to cross-encoder if bge failed and not specifically requested
            if self.model is None and self.use_model and model_name != "bge-reranker-large":
                try:
                    from sentence_transformers import CrossEncoder
                    fallback_model = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    self.model = CrossEncoder(fallback_model)
                    self.model_type = "cross-encoder"
                    print(f"✅ Loaded cross-encoder model: {fallback_model}")
                except Exception as e:
                    print(f"⚠️ Could not load cross-encoder model: {e}")
                    self.use_model = False

            # Final fallback to feature-based
            if self.model is None:
                print("   Falling back to feature-based reranking")
                self.use_model = False
                self.model_type = "feature"

    def rerank(self,
               query: str,
               results: List[RetrievalResult],
               top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Rerank retrieval results

        Args:
            query: Original query
            results: List of retrieval results
            top_k: Optional limit on number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        # Use model-based or feature-based reranking
        if self.use_model and self.model is not None:
            results = self._model_rerank(query, results)
        else:
            results = self._feature_rerank(query, results)

        # Sort by rerank score
        results = sorted(results, key=lambda x: x.rerank_score, reverse=True)

        # Return top-k if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def _model_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Model-based reranking using BGE or cross-encoder"""
        # Prepare pairs
        pairs = [[query, result.content] for result in results]

        # Get scores based on model type
        if self.model_type == "bge":
            # BGE FlagReranker uses compute_score()
            scores = self.model.compute_score(pairs)
            # Handle single result case (returns scalar instead of list)
            if not isinstance(scores, list):
                scores = [scores]
        else:
            # Cross-encoder uses predict()
            scores = self.model.predict(pairs)

        # Update rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)

        return results

    def _feature_rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Feature-based reranking"""
        for idx, result in enumerate(results):
            features = self._extract_features(query, result, position=idx)
            result.rerank_score = self._calculate_score(features, result)

        return results

    def _extract_features(self, query: str, result: RetrievalResult, position: int = 0) -> Dict[str, float]:
        """Extract reranking features"""
        query_words = set(query.lower().split())
        content_words = set(result.content.lower().split())

        features = {
            # Query-document overlap
            'query_overlap': (
                len(query_words.intersection(content_words)) / len(query_words)
                if query_words else 0
            ),

            # Section relevance
            'section_relevance': self._get_section_score(
                result.metadata.get('section_type', 'content')
            ),

            # Content quality indicators
            'content_quality': self._assess_quality(result.metadata),

            # Length penalty (prefer moderate length)
            'length_penalty': self._length_penalty(len(result.content)),

            # Has formulas (valuable for technical content)
            'has_formulas': 1.0 if result.metadata.get('has_formulas', False) else 0.0,

            # Has citations (indicates authoritative content)
            'has_citations': 1.0 if result.metadata.get('has_citations', False) else 0.0,

            # Position penalty (slight preference for earlier results)
            'position_penalty': max(0.0, 1.0 - (position * 0.05))
        }

        return features

    def _get_section_score(self, section_type: str) -> float:
        """Get relevance score for section type"""
        return self.section_relevance.get(section_type, 0.5)

    def _assess_quality(self, metadata: Dict) -> float:
        """Assess content quality from metadata"""
        quality = 0.5  # Base score

        # Bonus for rich content
        if metadata.get('has_formulas', False):
            quality += 0.15
        if metadata.get('has_code', False):
            quality += 0.10
        if metadata.get('has_citations', False):
            quality += 0.15
        if metadata.get('has_table', False):
            quality += 0.10

        return min(quality, 1.0)

    def _length_penalty(self, length: int) -> float:
        """Calculate length penalty"""
        optimal_length = 500  # Ideal chunk length

        if length < 100:
            return 0.3  # Too short
        elif length > 1500:
            return 0.6  # Too long
        elif 200 <= length <= 800:
            return 1.0  # Optimal range
        else:
            # Gradual penalty
            deviation = abs(length - optimal_length) / optimal_length
            return max(0.5, 1.0 - deviation * 0.5)

    def _calculate_score(self, features: Dict[str, float], result: RetrievalResult) -> float:
        """Calculate final rerank score"""
        # Weighted combination of features
        feature_score = sum(
            features.get(feature, 0.0) * weight
            for feature, weight in self.feature_weights.items()
        )

        # Combine with original retrieval score (if available)
        original_score = max(result.vector_score, result.combined_score)

        # Final score: 70% features + 30% original retrieval score
        final_score = 0.7 * feature_score + 0.3 * original_score

        return final_score


# ============================================================================
# MMR Diversifier
# ============================================================================

class MMRDiversifier:
    """
    Maximal Marginal Relevance (MMR) for result diversification

    Balances relevance and diversity to avoid redundant results.
    """

    def __init__(self, lambda_param: float = None):
        """
        Initialize MMR diversifier

        Args:
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
                         Default from config or 0.7
        """
        if USE_CONFIG and lambda_param is None:
            # Note: config.retrieval.mmr_diversity is actually the diversity weight
            # So lambda = 1 - diversity
            self.lambda_param = 1.0 - config.retrieval.mmr_diversity
        else:
            self.lambda_param = lambda_param or 0.7

        print(f"🎯 MMR Diversifier initialized (λ={self.lambda_param})")

    def diversify(self,
                  results: List[RetrievalResult],
                  top_k: int = 5,
                  similarity_threshold: float = 0.85) -> List[RetrievalResult]:
        """
        Apply MMR to diversify results

        Args:
            results: List of retrieval results
            top_k: Number of diverse results to return
            similarity_threshold: Threshold for considering results similar

        Returns:
            Diversified results
        """
        if not results or len(results) <= top_k:
            return results

        # Initialize selected and remaining
        selected = [results[0]]  # Start with highest-scoring result
        remaining = results[1:]

        # Iteratively select diverse results
        while len(selected) < top_k and remaining:
            best_idx = -1
            best_mmr_score = -float('inf')

            # Calculate MMR score for each remaining result
            for idx, candidate in enumerate(remaining):
                # Relevance score
                relevance = self._get_relevance_score(candidate)

                # Maximum similarity to selected results
                max_similarity = max(
                    self._calculate_similarity(candidate, selected_result)
                    for selected_result in selected
                )

                # MMR formula: λ * relevance - (1-λ) * max_similarity
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * max_similarity
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx

            # Add best result to selected
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
            else:
                break

        return selected

    def _get_relevance_score(self, result: RetrievalResult) -> float:
        """Get relevance score from result"""
        # Use the highest available score
        return max(
            result.rerank_score,
            result.combined_score,
            result.vector_score
        )

    def _calculate_similarity(self, result1: RetrievalResult, result2: RetrievalResult) -> float:
        """Calculate similarity between two results"""
        # Simple word-level Jaccard similarity
        words1 = set(result1.content.lower().split())
        words2 = set(result2.content.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


# ============================================================================
# Relevance Filter
# ============================================================================

class RelevanceFilter:
    """Filter results by relevance threshold with minimum keep guarantee"""

    def __init__(self, threshold: float = None, min_keep_k: int = 3):
        """
        Initialize relevance filter

        Args:
            threshold: Minimum score threshold (default from config or 0.3)
            min_keep_k: Minimum number of results to keep regardless of threshold
        """
        if USE_CONFIG and threshold is None:
            self.threshold = config.retrieval.similarity_threshold
        else:
            self.threshold = threshold or 0.3  # Lowered from 0.5

        self.min_keep_k = min_keep_k

        print(f"🔍 Relevance Filter initialized (threshold={self.threshold}, min_keep={min_keep_k})")

    def filter(self, results: List[RetrievalResult], min_keep: int = None) -> List[RetrievalResult]:
        """
        Filter results below threshold, but keep at least min_keep results

        Args:
            results: Results to filter
            min_keep: Override minimum keep count (default uses self.min_keep_k)

        Returns:
            Filtered results, guaranteed to have at least min_keep if available
        """
        min_keep = min_keep if min_keep is not None else self.min_keep_k

        # Score all results
        scored_results = []
        for result in results:
            # Get the highest score available
            max_score = max(
                result.rerank_score,
                result.combined_score,
                result.vector_score
            )
            scored_results.append((max_score, result))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)

        # Filter by threshold
        filtered = [r for score, r in scored_results if score >= self.threshold]

        # Guarantee minimum results
        if len(filtered) < min_keep and len(scored_results) > 0:
            # Keep top min_keep results regardless of threshold
            filtered = [r for _, r in scored_results[:min_keep]]
            print(f"⚠️  Only {len([s for s, _ in scored_results if s >= self.threshold])} above threshold, keeping top {len(filtered)}")

        print(f"✅ Filtered: {len(results)} → {len(filtered)} results (threshold={self.threshold})")
        return filtered


# ============================================================================
# Deduplicator
# ============================================================================

class Deduplicator:
    """Remove duplicate or near-duplicate results"""

    def __init__(self, similarity_threshold: float = 0.9):
        """
        Initialize deduplicator

        Args:
            similarity_threshold: Threshold for considering results duplicates
        """
        self.similarity_threshold = similarity_threshold

    def deduplicate(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Remove duplicates from results"""
        if not results:
            return []

        # Keep track of unique results
        unique_results = []
        seen_contents = set()

        for result in results:
            # Simple deduplication by content hash
            content_key = result.content[:200]  # First 200 chars

            # Check for exact duplicates
            if content_key in seen_contents:
                continue

            # Check for near-duplicates
            is_duplicate = False
            for unique_result in unique_results:
                similarity = self._calculate_similarity(result.content, unique_result.content)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)
                seen_contents.add(content_key)

        print(f"✅ Deduplicated: {len(results)} → {len(unique_results)} results")
        return unique_results

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0


# ============================================================================
# Composite Ranker
# ============================================================================

class CompositeRanker:
    """
    Composite ranker that applies multiple ranking/filtering operations

    Pipeline: Rerank → Filter → Diversify → Deduplicate
    """

    def __init__(self,
                 enable_reranking: bool = True,
                 enable_filtering: bool = True,
                 enable_diversification: bool = True,
                 enable_deduplication: bool = True):
        """
        Initialize composite ranker

        Args:
            enable_reranking: Enable cross-encoder reranking
            enable_filtering: Enable relevance filtering
            enable_diversification: Enable MMR diversification
            enable_deduplication: Enable deduplication
        """
        self.enable_reranking = enable_reranking
        self.enable_filtering = enable_filtering
        self.enable_diversification = enable_diversification
        self.enable_deduplication = enable_deduplication

        # Initialize components
        if enable_reranking:
            self.reranker = CrossEncoderRanker(use_model=True)  # Try bge-reranker-large by default

        if enable_filtering:
            self.filter = RelevanceFilter()

        if enable_diversification:
            self.diversifier = MMRDiversifier()

        if enable_deduplication:
            self.deduplicator = Deduplicator()

        print("🎨 Composite Ranker initialized")
        print(f"   - Reranking: {enable_reranking}")
        print(f"   - Filtering: {enable_filtering}")
        print(f"   - Diversification: {enable_diversification}")
        print(f"   - Deduplication: {enable_deduplication}")

    def process(self,
                query: str,
                results: List[RetrievalResult],
                top_k: int = 5) -> List[RetrievalResult]:
        """
        Apply full ranking pipeline

        Args:
            query: Original query
            results: Retrieval results
            top_k: Number of final results

        Returns:
            Processed results
        """
        print(f"📊 Processing {len(results)} results...")

        # 1. Reranking
        if self.enable_reranking:
            results = self.reranker.rerank(query, results)
            print(f"   ✓ Reranked")

        # 2. Filtering (with min_keep guarantee)
        if self.enable_filtering:
            # Guarantee we keep at least top_k results
            results = self.filter.filter(results, min_keep=top_k)
            print(f"   ✓ Filtered ({len(results)} remain)")

        # 3. Diversification
        if self.enable_diversification:
            results = self.diversifier.diversify(results, top_k=top_k)
            print(f"   ✓ Diversified ({len(results)} results)")

        # 4. Deduplication
        if self.enable_deduplication:
            results = self.deduplicator.deduplicate(results)
            print(f"   ✓ Deduplicated ({len(results)} results)")

        # Final top-k selection
        results = results[:top_k]

        print(f"✅ Final: {len(results)} results")
        return results


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'CrossEncoderRanker',
    'MMRDiversifier',
    'RelevanceFilter',
    'Deduplicator',
    'CompositeRanker'
]

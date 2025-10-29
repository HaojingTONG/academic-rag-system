"""
Answer Quality Enhancement Module
==================================

This module provides:
1. Post-generation summary injection
2. Faithfulness checking (verify answer matches context)
3. Answer quality metrics

Usage:
    from rag.answer_quality import AnswerEnhancer, FaithfulnessChecker

    # Enhance answer with summary
    enhancer = AnswerEnhancer()
    enhanced = enhancer.add_summary(answer, question, sources)

    # Check faithfulness
    checker = FaithfulnessChecker()
    score = checker.check_faithfulness(answer, context)
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FaithfulnessScore:
    """Faithfulness assessment result"""
    score: float  # 0.0 to 1.0
    citation_coverage: float  # Ratio of answer sentences with citations
    claim_support: float  # Ratio of claims supported by context
    verbatim_copying: float  # Ratio of verbatim copying (lower is better)
    keyword_coverage: float  # Ratio of required keywords present
    issues: List[str]  # List of faithfulness issues found


@dataclass
class KeywordValidation:
    """Keyword-based validation result"""
    required_present: List[str]
    required_missing: List[str]
    optional_present: List[str]
    coverage: float  # Ratio of required keywords present
    needs_refinement: bool  # True if missing critical keywords


@dataclass
class QualityMetrics:
    """Answer quality metrics"""
    word_count: int
    sentence_count: int
    citation_count: int
    unique_citations: int
    has_summary: bool
    technical_terms: int
    readability_score: float


class KeywordValidator:
    """Validate presence of required keywords in answer"""

    # Concept-specific keyword requirements
    CONCEPT_KEYWORDS = {
        'transformer': {
            'required': ['self-attention', 'encoder-decoder', 'attention mechanism', 'parallel'],
            'optional': ['positional encoding', 'multi-head', 'recurrence', 'sequence', 'vaswani']
        },
        'bert': {
            'required': ['bidirectional', 'transformer', 'pre-training', 'masked'],
            'optional': ['language model', 'fine-tuning', 'bert', 'encoding']
        },
        'attention': {
            'required': ['attention mechanism', 'weight', 'relevance'],
            'optional': ['query', 'key', 'value', 'self-attention', 'alignment']
        },
        'gpt': {
            'required': ['generative', 'autoregressive', 'transformer'],
            'optional': ['language model', 'pre-training', 'decoder', 'causal']
        }
    }

    def detect_concept(self, question: str) -> Optional[str]:
        """
        Detect which concept the question is about

        Args:
            question: User question

        Returns:
            Concept name or None
        """
        question_lower = question.lower()

        for concept in self.CONCEPT_KEYWORDS.keys():
            if concept in question_lower:
                return concept

        return None

    def validate_keywords(
        self,
        answer: str,
        concept: Optional[str] = None,
        custom_required: Optional[List[str]] = None,
        custom_optional: Optional[List[str]] = None
    ) -> KeywordValidation:
        """
        Validate keyword presence in answer

        Args:
            answer: Generated answer
            concept: Concept name (e.g., 'transformer', 'bert')
            custom_required: Custom required keywords
            custom_optional: Custom optional keywords

        Returns:
            KeywordValidation object
        """
        # Get keyword lists
        if custom_required is not None:
            required_keywords = custom_required
            optional_keywords = custom_optional or []
        elif concept and concept in self.CONCEPT_KEYWORDS:
            required_keywords = self.CONCEPT_KEYWORDS[concept]['required']
            optional_keywords = self.CONCEPT_KEYWORDS[concept]['optional']
        else:
            # No validation criteria
            return KeywordValidation(
                required_present=[],
                required_missing=[],
                optional_present=[],
                coverage=1.0,
                needs_refinement=False
            )

        answer_lower = answer.lower()

        # Check required keywords
        required_present = [kw for kw in required_keywords if kw.lower() in answer_lower]
        required_missing = [kw for kw in required_keywords if kw.lower() not in answer_lower]

        # Check optional keywords
        optional_present = [kw for kw in optional_keywords if kw.lower() in answer_lower]

        # Calculate coverage
        coverage = len(required_present) / len(required_keywords) if required_keywords else 1.0

        # Determine if refinement needed (missing >50% of required keywords)
        needs_refinement = coverage < 0.5

        return KeywordValidation(
            required_present=required_present,
            required_missing=required_missing,
            optional_present=optional_present,
            coverage=coverage,
            needs_refinement=needs_refinement
        )


class FaithfulnessChecker:
    """Check if generated answer is faithful to the source context"""

    def __init__(self, verbatim_threshold: int = 5):
        """
        Initialize faithfulness checker

        Args:
            verbatim_threshold: Min consecutive words to count as verbatim copy
        """
        self.verbatim_threshold = verbatim_threshold
        self.keyword_validator = KeywordValidator()

    def check_faithfulness(
        self,
        answer: str,
        context: str,
        question: Optional[str] = None,
        min_citation_ratio: float = 0.5,
        required_keywords: Optional[List[str]] = None
    ) -> FaithfulnessScore:
        """
        Check answer faithfulness to context

        Args:
            answer: Generated answer
            context: Source context
            question: Original question (for concept detection)
            min_citation_ratio: Minimum ratio of sentences with citations
            required_keywords: Custom required keywords to check

        Returns:
            FaithfulnessScore object
        """
        issues = []

        # 1. Check citation coverage
        citation_coverage = self._check_citation_coverage(answer)
        if citation_coverage < min_citation_ratio:
            issues.append(
                f"Low citation coverage: {citation_coverage:.1%} "
                f"(threshold: {min_citation_ratio:.1%})"
            )

        # 2. Check for verbatim copying
        verbatim_ratio = self._check_verbatim_copying(answer, context)
        if verbatim_ratio > 0.3:  # More than 30% verbatim is concerning
            issues.append(
                f"High verbatim copying: {verbatim_ratio:.1%} of answer"
            )

        # 3. Check claim support (simplified)
        claim_support = self._check_claim_support(answer, context)
        # Note: Lower threshold because rephrased answers will have different words
        # This is actually a good sign of synthesis!
        if claim_support < 0.3:  # At least 30% keyword overlap
            issues.append(
                f"Weak claim support: {claim_support:.1%} "
                f"(threshold: 30%)"
            )

        # 4. Check keyword coverage (NEW)
        keyword_coverage = 1.0  # Default if no keywords to check
        if question or required_keywords:
            # Detect concept from question
            concept = None
            if question:
                concept = self.keyword_validator.detect_concept(question)

            # Validate keywords
            validation = self.keyword_validator.validate_keywords(
                answer,
                concept=concept,
                custom_required=required_keywords
            )

            keyword_coverage = validation.coverage

            if validation.needs_refinement:
                issues.append(
                    f"Missing required keywords: {', '.join(validation.required_missing)}"
                )

        # Calculate overall score
        overall_score = (
            citation_coverage * 0.3 +
            claim_support * 0.3 +
            (1 - verbatim_ratio) * 0.2 +
            keyword_coverage * 0.2
        )

        return FaithfulnessScore(
            score=overall_score,
            citation_coverage=citation_coverage,
            claim_support=claim_support,
            verbatim_copying=verbatim_ratio,
            keyword_coverage=keyword_coverage,
            issues=issues
        )

    def _check_citation_coverage(self, answer: str) -> float:
        """Check what ratio of sentences have citations"""
        # Split into sentences (simple approach)
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]

        if not sentences:
            return 0.0

        # Count sentences with citations [1], [2], etc.
        cited_sentences = 0
        for sentence in sentences:
            if re.search(r'\[\d+\]', sentence):
                cited_sentences += 1

        return cited_sentences / len(sentences)

    def _check_verbatim_copying(self, answer: str, context: str) -> float:
        """
        Check for verbatim copying from context

        Returns ratio of answer that is verbatim copied
        """
        # Normalize texts
        answer_words = answer.lower().split()
        context_words = context.lower().split()

        if not answer_words:
            return 0.0

        # Find consecutive word matches
        verbatim_words = 0

        for i in range(len(answer_words) - self.verbatim_threshold + 1):
            # Check if next N words appear consecutively in context
            phrase = ' '.join(answer_words[i:i + self.verbatim_threshold])

            if phrase in ' '.join(context_words):
                verbatim_words += self.verbatim_threshold

        return min(verbatim_words / len(answer_words), 1.0)

    def _check_claim_support(self, answer: str, context: str) -> float:
        """
        Check if claims in answer are supported by context

        Simple approach: check keyword overlap
        """
        # Extract technical terms and keywords from answer
        answer_terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer))
        answer_terms.update(re.findall(r'\b\w{6,}\b', answer.lower()))

        if not answer_terms:
            return 1.0  # No claims to verify

        # Check how many appear in context
        context_lower = context.lower()
        supported = sum(1 for term in answer_terms if term.lower() in context_lower)

        return supported / len(answer_terms)


class AnswerEnhancer:
    """Enhance generated answers with summaries and improvements"""

    def add_summary(
        self,
        answer: str,
        question: str,
        sources: List[Dict],
        max_summary_length: int = 100
    ) -> str:
        """
        Add a concluding summary paragraph to the answer

        Args:
            answer: Generated answer
            question: Original question
            sources: Retrieved sources
            max_summary_length: Maximum summary length in words

        Returns:
            Enhanced answer with summary
        """
        # Check if answer already has a summary
        if self._has_summary(answer):
            return answer

        # Generate summary based on answer content
        summary = self._generate_summary(answer, question, sources)

        # Add summary at the end
        if summary:
            enhanced = answer.rstrip()
            if not enhanced.endswith(('.', '!', '?')):
                enhanced += '.'

            enhanced += f"\n\n**Summary:**\n{summary}"
            return enhanced

        return answer

    def _has_summary(self, answer: str) -> bool:
        """Check if answer already has a summary section"""
        return bool(re.search(r'\*\*Summary[:\s]', answer, re.IGNORECASE))

    def _generate_summary(
        self,
        answer: str,
        question: str,
        sources: List[Dict]
    ) -> str:
        """
        Generate a concise summary of the answer

        Args:
            answer: Generated answer
            question: Original question
            sources: Retrieved sources

        Returns:
            Summary text
        """
        # Extract main concept from question
        concept = self._extract_concept(question)

        # Count citations
        citations = set(re.findall(r'\[(\d+)\]', answer))
        num_sources = len(citations)

        # Get source titles
        source_titles = [s.get('title', 'Unknown')[:40] for s in sources[:3]]

        # Generate summary
        if num_sources == 0:
            summary = f"This answer addresses {concept} based on available research."
        elif num_sources == 1:
            summary = f"In summary, {concept} is explained drawing from {source_titles[0]}."
        else:
            summary = (
                f"In summary, {concept} is a significant concept in the field, "
                f"as demonstrated across {num_sources} research papers including "
                f"{source_titles[0]} and others."
            )

        return summary

    def _extract_concept(self, question: str) -> str:
        """Extract main concept from question"""
        # Remove question words
        question_lower = question.lower()
        for word in ['what is', 'what are', 'explain', 'describe', 'define']:
            question_lower = question_lower.replace(word, '')

        # Extract remaining key phrase
        words = question_lower.strip(' ?').split()
        if words:
            # Take first 3-4 words as concept
            concept = ' '.join(words[:min(4, len(words))])
            return concept.strip()

        return "the concept"

    def enhance_technical_clarity(self, answer: str) -> str:
        """
        Enhance technical clarity of answer

        Args:
            answer: Generated answer

        Returns:
            Enhanced answer with better technical clarity
        """
        # This is a placeholder for more sophisticated enhancements
        # Could include:
        # - Ensure technical terms are properly introduced
        # - Add clarifying parenthetical explanations
        # - Improve transition sentences

        # For now, just ensure proper formatting
        enhanced = answer

        # Ensure sections are properly spaced
        enhanced = re.sub(r'\n{3,}', '\n\n', enhanced)

        # Ensure citations have proper spacing
        enhanced = re.sub(r'(\w)\[', r'\1 [', enhanced)

        return enhanced


class QualityAnalyzer:
    """Analyze answer quality metrics"""

    def analyze(self, answer: str) -> QualityMetrics:
        """
        Analyze answer quality

        Args:
            answer: Generated answer

        Returns:
            QualityMetrics object
        """
        # Count words and sentences
        words = answer.split()
        sentences = [s for s in re.split(r'[.!?]+', answer) if s.strip()]

        # Count citations
        citations = re.findall(r'\[(\d+)\]', answer)
        unique_citations = set(citations)

        # Check for summary
        has_summary = bool(re.search(r'\*\*Summary[:\s]', answer, re.IGNORECASE))

        # Count technical terms (simplified: words with capital letters)
        technical_terms = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer))

        # Simple readability score (average sentence length)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        # Normalize to 0-1 scale (15 words is ideal, 30+ is complex)
        readability = max(0, min(1, 1 - abs(avg_sentence_length - 15) / 15))

        return QualityMetrics(
            word_count=len(words),
            sentence_count=len(sentences),
            citation_count=len(citations),
            unique_citations=len(unique_citations),
            has_summary=has_summary,
            technical_terms=technical_terms,
            readability_score=readability
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def check_answer_quality(
    answer: str,
    context: str,
    question: str,
    sources: List[Dict]
) -> Dict:
    """
    Comprehensive answer quality check

    Args:
        answer: Generated answer
        context: Source context
        question: Original question
        sources: Retrieved sources

    Returns:
        Dictionary with quality metrics and faithfulness score
    """
    # Check faithfulness
    faithfulness_checker = FaithfulnessChecker()
    faithfulness = faithfulness_checker.check_faithfulness(answer, context)

    # Analyze quality
    quality_analyzer = QualityAnalyzer()
    quality = quality_analyzer.analyze(answer)

    return {
        'faithfulness': {
            'score': faithfulness.score,
            'citation_coverage': faithfulness.citation_coverage,
            'claim_support': faithfulness.claim_support,
            'verbatim_copying': faithfulness.verbatim_copying,
            'issues': faithfulness.issues
        },
        'quality': {
            'word_count': quality.word_count,
            'sentence_count': quality.sentence_count,
            'citation_count': quality.citation_count,
            'unique_citations': quality.unique_citations,
            'has_summary': quality.has_summary,
            'technical_terms': quality.technical_terms,
            'readability_score': quality.readability_score
        },
        'overall_score': (faithfulness.score + quality.readability_score) / 2
    }


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'FaithfulnessScore',
    'KeywordValidation',
    'QualityMetrics',
    'KeywordValidator',
    'FaithfulnessChecker',
    'AnswerEnhancer',
    'QualityAnalyzer',
    'check_answer_quality'
]

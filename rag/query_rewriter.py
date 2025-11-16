"""
Query Rewriter - Multi-Query Expansion for Better Retrieval
===========================================================

Implements query rewriting strategies to improve retrieval quality:
1. Semantic expansion (rephrasing)
2. Multi-perspective queries
3. Term expansion
4. Question decomposition
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
import re


@dataclass
class QueryIntent:
    """Detected query intent"""
    intent_type: str  # definition, comparison, method, recent_work, general
    is_general: bool  # True if asking for general knowledge (e.g., "What is AI?")
    is_specific: bool  # True if asking about specific paper/method
    key_concepts: List[str]  # Extracted key concepts
    confidence: float  # Intent detection confidence


class QueryRewriter:
    """
    Rewrites user queries to improve retrieval quality

    Strategies:
    - Multi-query expansion (generate diverse queries)
    - Term expansion (add synonyms)
    - Question decomposition (break complex queries)
    """

    # General knowledge patterns
    GENERAL_PATTERNS = [
        r'^what\s+(is|are)\s+(the\s+)?(\w+)\??$',  # "What is AI?"
        r'^(define|explain|introduce)\s+',  # "Define machine learning"
        r'^give\s+me\s+an?\s+(definition|overview|introduction)',
        r'^tell\s+me\s+about\s+(\w+)$'
    ]

    # Specific research patterns
    SPECIFIC_PATTERNS = [
        r'(paper|research|study|work)\s+(about|on)',
        r'(author|researcher)\s+',
        r'(citation|reference)',
        r'in\s+the\s+(paper|work|study)',
        r'how\s+does\s+(\w+)\s+(work|perform|compare)'
    ]

    def __init__(self):
        """Initialize query rewriter"""
        self.concept_synonyms = {
            'AI': ['artificial intelligence', 'machine intelligence', 'intelligent systems'],
            'ML': ['machine learning', 'statistical learning', 'automated learning'],
            'NLP': ['natural language processing', 'text processing', 'language understanding'],
            'CV': ['computer vision', 'visual recognition', 'image analysis'],
            'transformer': ['attention mechanism', 'self-attention', 'transformer architecture'],
            'neural network': ['deep learning', 'neural model', 'deep neural network'],
            'CNN': ['convolutional network', 'convnet', 'convolutional neural network'],
            'RNN': ['recurrent network', 'recurrent neural network', 'sequence model'],
            'BERT': ['bidirectional transformer', 'masked language model'],
            'GPT': ['generative pre-trained transformer', 'autoregressive language model'],
        }

    def detect_intent(self, query: str) -> QueryIntent:
        """
        Detect query intent and characteristics

        Args:
            query: User query

        Returns:
            QueryIntent with detected characteristics
        """
        query_lower = query.lower().strip()

        # Check if general knowledge question
        is_general = any(re.search(pattern, query_lower) for pattern in self.GENERAL_PATTERNS)

        # Check if specific research question
        is_specific = any(re.search(pattern, query_lower) for pattern in self.SPECIFIC_PATTERNS)

        # Detect intent type
        if is_general and not is_specific:
            intent_type = 'definition'
            confidence = 0.9
        elif 'compare' in query_lower or 'difference' in query_lower:
            intent_type = 'comparison'
            confidence = 0.8
        elif 'how' in query_lower and ('work' in query_lower or 'implement' in query_lower):
            intent_type = 'method'
            confidence = 0.8
        elif 'recent' in query_lower or 'latest' in query_lower or 'state-of-the-art' in query_lower:
            intent_type = 'recent_work'
            confidence = 0.8
        else:
            intent_type = 'general'
            confidence = 0.5

        # Extract key concepts
        key_concepts = self._extract_concepts(query)

        return QueryIntent(
            intent_type=intent_type,
            is_general=is_general,
            is_specific=is_specific,
            key_concepts=key_concepts,
            confidence=confidence
        )

    def _extract_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        # Simple keyword extraction (can be enhanced with NER)
        query_lower = query.lower()

        concepts = []

        # Check for known concepts
        for concept in self.concept_synonyms.keys():
            if concept.lower() in query_lower:
                concepts.append(concept)

        # Extract capitalized terms (likely acronyms/names)
        words = query.split()
        for word in words:
            # Remove punctuation
            word_clean = re.sub(r'[^\w\s-]', '', word)
            if word_clean.isupper() and len(word_clean) > 1:
                concepts.append(word_clean)

        return list(set(concepts))

    def expand_query(self, query: str, max_queries: int = 3) -> List[str]:
        """
        Generate multiple query variations for better retrieval

        Args:
            query: Original query
            max_queries: Maximum number of expanded queries

        Returns:
            List of query variations (including original)
        """
        intent = self.detect_intent(query)

        expanded = [query]  # Always include original

        # Strategy 1: For general questions, create specific academic queries
        if intent.is_general:
            expanded.extend(self._expand_general_query(query, intent))

        # Strategy 2: Add synonym-based variations
        expanded.extend(self._expand_with_synonyms(query, intent.key_concepts))

        # Strategy 3: Add perspective variations
        expanded.extend(self._add_perspective_variations(query, intent))

        # Remove duplicates and limit
        unique_queries = []
        seen = set()
        for q in expanded:
            q_normalized = q.lower().strip()
            if q_normalized not in seen:
                unique_queries.append(q)
                seen.add(q_normalized)
                if len(unique_queries) >= max_queries:
                    break

        return unique_queries

    def _expand_general_query(self, query: str, intent: QueryIntent) -> List[str]:
        """
        Expand general questions into specific academic queries

        Example:
        "What is AI?" →
        - "artificial intelligence definition and applications"
        - "AI fundamental concepts and techniques"
        - "introduction to artificial intelligence systems"
        """
        expanded = []

        # Extract the main concept
        query_lower = query.lower()

        # Pattern: "What is X?" → academic variations
        match = re.search(r'what\s+is\s+(the\s+)?(.+?)\??$', query_lower)
        if match:
            concept = match.group(2).strip()

            # Get expanded form if acronym
            concept_expanded = concept
            for key, synonyms in self.concept_synonyms.items():
                if concept == key.lower():
                    concept_expanded = synonyms[0]  # Use first synonym
                    break

            expanded.extend([
                f"{concept_expanded} definition and fundamentals",
                f"introduction to {concept_expanded}",
                f"{concept_expanded} key concepts and techniques",
                f"{concept_expanded} overview and applications"
            ])

        # Pattern: "Define X" → variations
        match = re.search(r'(define|explain|introduce)\s+(.+?)$', query_lower)
        if match:
            concept = match.group(2).strip()
            expanded.extend([
                f"{concept} definition",
                f"{concept} fundamental concepts",
                f"what is {concept}"
            ])

        return expanded[:2]  # Return top 2 variations

    def _expand_with_synonyms(self, query: str, concepts: List[str]) -> List[str]:
        """Expand query by replacing concepts with synonyms"""
        expanded = []

        for concept in concepts:
            if concept in self.concept_synonyms:
                for synonym in self.concept_synonyms[concept][:2]:  # Use top 2 synonyms
                    # Replace concept with synonym
                    variant = re.sub(
                        r'\b' + re.escape(concept) + r'\b',
                        synonym,
                        query,
                        flags=re.IGNORECASE
                    )
                    if variant != query:
                        expanded.append(variant)

        return expanded[:2]  # Return top 2

    def _add_perspective_variations(self, query: str, intent: QueryIntent) -> List[str]:
        """Add different perspective variations of the query"""
        expanded = []

        if intent.intent_type == 'definition':
            # Add "how it works" perspective
            for concept in intent.key_concepts:
                expanded.append(f"how does {concept} work")
                expanded.append(f"{concept} architecture and design")

        elif intent.intent_type == 'comparison':
            # Add "advantages and disadvantages" perspective
            expanded.append(query.replace("compare", "advantages and disadvantages of"))

        elif intent.intent_type == 'method':
            # Add "implementation" perspective
            expanded.append(query.replace("how", "implementation of"))

        return expanded[:1]  # Return top 1


# Convenience function for pipeline integration
def expand_query(query: str, max_queries: int = 3) -> List[str]:
    """
    Expand query into multiple variations

    Args:
        query: Original query
        max_queries: Maximum number of queries to return

    Returns:
        List of expanded queries
    """
    rewriter = QueryRewriter()
    return rewriter.expand_query(query, max_queries)


def detect_query_intent(query: str) -> QueryIntent:
    """
    Detect query intent

    Args:
        query: User query

    Returns:
        QueryIntent object
    """
    rewriter = QueryRewriter()
    return rewriter.detect_intent(query)

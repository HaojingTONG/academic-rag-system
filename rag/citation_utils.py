"""
Citation Utilities - Citation Extraction and Sanitization
=========================================================

This module provides utilities for:
- Extracting citation markers from generated text
- Mapping citations to source documents
- Sanitizing sources (removing unreferenced sources)
- Validating citation consistency
"""

import re
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass


@dataclass
class CitationMatch:
    """A citation reference found in text"""
    marker: str  # e.g., "[1]"
    number: int  # e.g., 1
    position: int  # Character position in text


class CitationExtractor:
    """Extract and analyze citations from generated text"""

    @staticmethod
    def extract_citations(text: str) -> List[CitationMatch]:
        """
        Extract all citation markers from text

        Args:
            text: Generated answer text

        Returns:
            List of CitationMatch objects

        Examples:
            >>> extract_citations("The transformer [1] uses attention [2].")
            [CitationMatch(marker='[1]', number=1, position=16),
             CitationMatch(marker='[2]', number=2, position=39)]
        """
        citations = []

        # Find all citation markers like [1], [2], etc.
        for match in re.finditer(r'\[(\d+)\]', text):
            citations.append(CitationMatch(
                marker=match.group(0),
                number=int(match.group(1)),
                position=match.start()
            ))

        return citations

    @staticmethod
    def get_cited_numbers(text: str) -> Set[int]:
        """
        Get set of all citation numbers used in text

        Args:
            text: Generated answer text

        Returns:
            Set of citation numbers

        Examples:
            >>> get_cited_numbers("The transformer [1] uses attention [2] and [1].")
            {1, 2}
        """
        citations = CitationExtractor.extract_citations(text)
        return {c.number for c in citations}

    @staticmethod
    def count_citations(text: str) -> Dict[int, int]:
        """
        Count how many times each citation is used

        Args:
            text: Generated answer text

        Returns:
            Dictionary mapping citation number to count

        Examples:
            >>> count_citations("The transformer [1] uses [2] and [1].")
            {1: 2, 2: 1}
        """
        citations = CitationExtractor.extract_citations(text)
        counts = {}
        for citation in citations:
            counts[citation.number] = counts.get(citation.number, 0) + 1
        return counts


class CitationSanitizer:
    """Sanitize citations and sources for consistency"""

    @staticmethod
    def validate_citations(text: str, num_sources: int) -> Tuple[bool, List[str]]:
        """
        Validate that all citations reference valid sources

        Args:
            text: Generated answer text
            num_sources: Number of available source documents

        Returns:
            Tuple of (is_valid, list of issues)

        Examples:
            >>> validate_citations("Uses [1] and [2].", num_sources=1)
            (False, ["Citation [2] references source 2 but only 1 sources available"])
        """
        cited_numbers = CitationExtractor.get_cited_numbers(text)
        issues = []
        is_valid = True

        # Check for citations beyond available sources
        max_cited = max(cited_numbers) if cited_numbers else 0
        if max_cited > num_sources:
            is_valid = False
            invalid = [n for n in cited_numbers if n > num_sources]
            issues.append(
                f"Citations {invalid} reference sources beyond available range [1-{num_sources}]"
            )

        # Check for citation [0] or negative
        invalid_nums = [n for n in cited_numbers if n <= 0]
        if invalid_nums:
            is_valid = False
            issues.append(f"Invalid citation numbers: {invalid_nums} (must be ≥1)")

        return is_valid, issues

    @staticmethod
    def sanitize_sources(answer: str, sources: List[Dict]) -> List[Dict]:
        """
        Remove sources that are not cited in the answer

        This reduces clutter and ensures only referenced sources are shown.

        Args:
            answer: Generated answer text
            sources: List of source documents (must have 'index' field)

        Returns:
            Filtered list of sources that are actually cited

        Examples:
            >>> sources = [
            ...     {'index': 1, 'title': 'Paper A'},
            ...     {'index': 2, 'title': 'Paper B'},
            ...     {'index': 3, 'title': 'Paper C'}
            ... ]
            >>> sanitize_sources("Based on [1] and [3].", sources)
            [{'index': 1, 'title': 'Paper A'}, {'index': 3, 'title': 'Paper C'}]
        """
        cited_numbers = CitationExtractor.get_cited_numbers(answer)

        if not cited_numbers:
            # No citations found - return empty list
            return []

        # Filter to only cited sources
        sanitized = [
            source for source in sources
            if source.get('index') in cited_numbers
        ]

        return sanitized

    @staticmethod
    def add_citation_warnings(answer: str, num_sources: int) -> str:
        """
        Add warning messages to answer if citations are invalid or missing

        Args:
            answer: Generated answer text
            num_sources: Number of available sources

        Returns:
            Answer with warnings appended (if needed)
        """
        cited_numbers = CitationExtractor.get_cited_numbers(answer)

        # No citations found
        if not cited_numbers and num_sources > 0:
            answer += "\n\n**Note:** This answer should cite specific sources. Please verify information independently."
            return answer

        # Validate citations
        is_valid, issues = CitationSanitizer.validate_citations(answer, num_sources)

        if not is_valid:
            answer += f"\n\n**Warning:** Citation issues detected:\n"
            for issue in issues:
                answer += f"- {issue}\n"
            answer += f"\nOnly sources [1]-[{num_sources}] are available."

        return answer

    @staticmethod
    def reindex_citations(answer: str, sources: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Reindex citations and sources to be sequential starting from 1

        Useful when sources have been filtered or reordered.

        Args:
            answer: Generated answer text
            sources: List of source documents with 'index' field

        Returns:
            Tuple of (updated_answer, updated_sources)

        Examples:
            >>> answer = "Based on [2] and [5]."
            >>> sources = [{'index': 2, 'title': 'A'}, {'index': 5, 'title': 'B'}]
            >>> reindex_citations(answer, sources)
            ("Based on [1] and [2].", [{'index': 1, 'title': 'A'}, {'index': 2, 'title': 'B'}])
        """
        # Build mapping from old index to new index
        old_to_new = {}
        new_sources = []

        for new_idx, source in enumerate(sources, 1):
            old_idx = source.get('index')
            old_to_new[old_idx] = new_idx

            # Update source index
            new_source = source.copy()
            new_source['index'] = new_idx
            new_sources.append(new_source)

        # Replace citations in answer
        new_answer = answer
        for old_idx, new_idx in sorted(old_to_new.items(), reverse=True):
            # Replace in reverse order to avoid conflicts (e.g., [1] -> [2] before [2] -> [3])
            new_answer = re.sub(
                rf'\[{old_idx}\]',
                f'[{new_idx}]',
                new_answer
            )

        return new_answer, new_sources


class CitationFormatter:
    """Format citations for display"""

    @staticmethod
    def format_source_list(sources: List[Dict], style: str = "academic") -> str:
        """
        Format sources as a reference list

        Args:
            sources: List of source documents
            style: Formatting style ('academic', 'compact', 'verbose')

        Returns:
            Formatted reference list
        """
        if not sources:
            return ""

        lines = []
        lines.append("\n## Sources\n")

        for source in sources:
            idx = source.get('index', '?')
            title = source.get('title', 'Unknown')

            if style == "academic":
                # Academic format with metadata
                metadata = source.get('metadata', {})
                authors = metadata.get('authors', 'Unknown')
                year = metadata.get('year', 'n.d.')
                lines.append(f"[{idx}] {authors} ({year}). {title}")

            elif style == "compact":
                # Compact format
                lines.append(f"[{idx}] {title}")

            elif style == "verbose":
                # Verbose format with content preview
                content = source.get('content', '')[:200]
                lines.append(f"[{idx}] {title}")
                lines.append(f"    {content}...")
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def get_citation_stats(answer: str, sources: List[Dict]) -> Dict:
        """
        Get statistics about citations in answer

        Args:
            answer: Generated answer text
            sources: List of source documents

        Returns:
            Dictionary with citation statistics
        """
        cited_numbers = CitationExtractor.get_cited_numbers(answer)
        citation_counts = CitationExtractor.count_citations(answer)

        return {
            'total_sources': len(sources),
            'cited_sources': len(cited_numbers),
            'uncited_sources': len(sources) - len(cited_numbers),
            'total_citations': sum(citation_counts.values()),
            'unique_citations': len(cited_numbers),
            'citation_counts': citation_counts,
            'has_citations': len(cited_numbers) > 0,
            'citation_rate': len(cited_numbers) / len(sources) if sources else 0
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def extract_citations(text: str) -> List[CitationMatch]:
    """Convenience function to extract citations"""
    return CitationExtractor.extract_citations(text)


def get_cited_numbers(text: str) -> Set[int]:
    """Convenience function to get cited numbers"""
    return CitationExtractor.get_cited_numbers(text)


def sanitize_sources(answer: str, sources: List[Dict]) -> List[Dict]:
    """Convenience function to sanitize sources"""
    return CitationSanitizer.sanitize_sources(answer, sources)


def add_citation_warnings(answer: str, num_sources: int) -> str:
    """Convenience function to add warnings"""
    return CitationSanitizer.add_citation_warnings(answer, num_sources)


def validate_citations(text: str, num_sources: int) -> Tuple[bool, List[str]]:
    """Convenience function to validate citations"""
    return CitationSanitizer.validate_citations(text, num_sources)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'CitationMatch',
    'CitationExtractor',
    'CitationSanitizer',
    'CitationFormatter',
    'extract_citations',
    'get_cited_numbers',
    'sanitize_sources',
    'add_citation_warnings',
    'validate_citations'
]

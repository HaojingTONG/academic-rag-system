"""
DEPRECATED: This module has been moved to rag.retriever

This compatibility shim will be removed in v2.1.0 (planned: 3 months from now).

Please update your imports:
    OLD: from src.retriever.vector_store import VectorStore
    NEW: from rag.retriever import VectorStore

    OLD: from src.retriever.advanced_retrieval import HybridRetriever
    NEW: from rag.retriever import HybridRetriever
"""

import warnings
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Show deprecation warning
warnings.warn(
    "\n"
    "=" * 70 + "\n"
    "⚠️  DEPRECATION WARNING\n"
    "=" * 70 + "\n"
    "The 'src.retriever' module is deprecated and will be removed in v2.1.0.\n"
    "\n"
    "Please update your imports:\n"
    "    OLD: from src.retriever import VectorStore\n"
    "    NEW: from rag.retriever import VectorStore\n"
    "\n"
    "See MIGRATION.md for full migration guide.\n"
    "=" * 70,
    DeprecationWarning,
    stacklevel=2
)

# Import from new location and re-export
try:
    from rag.retriever import (
        VectorStore,
        BM25Retriever,
        HybridRetriever,
        QueryAnalyzer,
        RetrievalResult,
        QueryIntent
    )
    from rag.ranker import (
        CrossEncoderRanker as CrossEncoderReranker,
        MMRDiversifier
    )

    # Alias for backward compatibility
    QueryUnderstanding = QueryAnalyzer

    __all__ = [
        'VectorStore',
        'BM25Retriever',
        'HybridRetriever',
        'QueryAnalyzer',
        'QueryUnderstanding',  # Deprecated alias
        'RetrievalResult',
        'QueryIntent',
        'CrossEncoderReranker',
        'MMRDiversifier'
    ]

except ImportError as e:
    warnings.warn(f"Failed to import from new location: {e}", ImportWarning)
    # Fallback to old imports
    from .vector_store import VectorStore
    from .advanced_retrieval import (
        HybridRetriever,
        QueryUnderstanding,
        BM25Retriever,
        CrossEncoderReranker,
        MMRDiversifier
    )

    __all__ = [
        'VectorStore',
        'HybridRetriever',
        'QueryUnderstanding',
        'BM25Retriever',
        'CrossEncoderReranker',
        'MMRDiversifier'
    ]

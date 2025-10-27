"""
RAG Module - Unified Retrieval-Augmented Generation
===================================================

This module consolidates all RAG functionality into a clean, unified API.

Quick Start:
    from rag import RAGPipeline

    # Initialize and run
    pipeline = RAGPipeline()
    pipeline.initialize()

    # Query
    result = pipeline.query("What is BERT?")
    print(result['answer'])

Components:
    - retriever: Vector search, BM25, hybrid retrieval
    - ranker: Reranking, diversification, filtering
    - pipeline: End-to-end RAG orchestration

Replaces:
    - src/retriever/*
    - Parts of src/generator/*
    - Parts of src/evaluation/*
"""

# Version
__version__ = '2.0.0'

# Core Pipeline
from .pipeline import RAGPipeline, RAGConfig

# Retrieval Components
from .retriever import (
    VectorStore,
    BM25Retriever,
    HybridRetriever,
    QueryAnalyzer,
    RetrievalResult,
    QueryIntent
)

# Ranking Components
from .ranker import (
    CrossEncoderRanker,
    MMRDiversifier,
    RelevanceFilter,
    Deduplicator,
    CompositeRanker
)

# Exports
__all__ = [
    # Pipeline
    'RAGPipeline',
    'RAGConfig',

    # Retrieval
    'VectorStore',
    'BM25Retriever',
    'HybridRetriever',
    'QueryAnalyzer',
    'RetrievalResult',
    'QueryIntent',

    # Ranking
    'CrossEncoderRanker',
    'MMRDiversifier',
    'RelevanceFilter',
    'Deduplicator',
    'CompositeRanker',
]

# Module metadata
__author__ = 'Academic RAG System Team'
__description__ = 'Unified RAG module for question answering over academic papers'

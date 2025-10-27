"""
App Module - Entry Points for Academic RAG System
=================================================

Provides CLI and Web API entry points.

Usage:
    # CLI
    python -m app.cli query "What is BERT?"
    python -m app.cli --help

    # Web API
    python -m app.main
    uvicorn app.main:app --reload
"""

__version__ = '2.0.0'
__author__ = 'Academic RAG System Team'

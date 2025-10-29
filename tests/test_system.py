#!/usr/bin/env python3
"""
Academic RAG System - System Test Script
========================================

Tests all major components to ensure the system is working correctly.

Usage:
    python test_system.py

With OpenAI API key:
    OPENAI_API_KEY=sk-your-key python test_system.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def print_header(title):
    """Print test section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_imports():
    """Test basic imports"""
    print_header("1. Testing Basic Imports")

    try:
        import yaml
        import requests
        from dataclasses import dataclass
        import chromadb
        from loguru import logger
        print("✅ All basic dependencies available")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print_header("2. Testing Configuration")

    try:
        from configs.config_loader import load_config, Config

        config = load_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   - System: {config.system.name} v{config.system.version}")
        print(f"   - LLM Backend: {config.generation.llm_backend}")
        print(f"   - Model: {config.generation.model}")
        print(f"   - OpenAI API Key: {'Set' if config.generation.openai_api_key else 'Not set'}")
        print(f"   - Top-K: {config.retrieval.top_k}")
        print(f"   - Embedding Model: {config.embedding.model}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_openai_client():
    """Test OpenAI client initialization"""
    print_header("3. Testing OpenAI Client")

    try:
        from src.generator.llm_client import OpenAIClient, create_llm_manager_from_config

        # Test direct client
        client = OpenAIClient()
        print(f"   - Client initialized")
        print(f"   - API Key set: {bool(client.api_key)}")
        print(f"   - Available: {client.is_available()}")

        # Test manager from config
        manager = create_llm_manager_from_config()
        print(f"   - Manager backend: {manager.backend}")
        print(f"   - Preferred model: {manager.preferred_model}")

        if manager.openai_available:
            print("✅ OpenAI client ready")
        else:
            print("⚠️  OpenAI client initialized but API not available")
            print("   (Set OPENAI_API_KEY environment variable to enable)")

        return True
    except Exception as e:
        print(f"❌ OpenAI client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store():
    """Test vector store"""
    print_header("4. Testing Vector Store")

    try:
        from rag.retriever import VectorStore

        # Check if DB exists
        db_path = Path('vector_db')
        if not db_path.exists():
            print("⚠️  Vector DB not found at 'vector_db/'")
            print("   Run indexing first: make index")
            return False

        # Initialize
        store = VectorStore()
        print(f"   - Vector store initialized")

        # Test search
        results = store.search("transformer architecture", top_k=3)
        print(f"   - Search returned {len(results)} results")

        if len(results) > 0:
            print(f"   - Top result: {results[0].metadata.get('title', 'Unknown')[:60]}...")
            print(f"   - Score: {results[0].vector_score:.3f}")
            print("✅ Vector store working correctly")
        else:
            print("⚠️  No results found - vector DB may be empty")

        return True
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rag_pipeline():
    """Test RAG pipeline"""
    print_header("5. Testing RAG Pipeline")

    try:
        from rag.pipeline import RAGPipeline
        from rag.retriever import VectorStore
        from src.generator.llm_client import create_llm_manager_from_config

        # Initialize components
        print("   Initializing components...")
        vector_store = VectorStore()
        llm_manager = create_llm_manager_from_config()

        # Initialize pipeline
        print("   Initializing pipeline...")
        pipeline = RAGPipeline()
        pipeline.initialize(
            vector_store=vector_store,
            llm_client=llm_manager
        )

        # Test query (retrieval only to avoid API costs)
        print("   Testing retrieval...")
        question = "What is the transformer architecture?"
        results = vector_store.search(question, top_k=3)

        if len(results) > 0:
            print(f"   - Retrieved {len(results)} relevant documents")
            print("✅ RAG Pipeline initialized successfully")
            print("\n   📝 To test full generation with OpenAI:")
            print("      export OPENAI_API_KEY=sk-your-key-here")
            print("      python -m app.cli query --interactive")
        else:
            print("⚠️  Pipeline initialized but no documents retrieved")

        return True
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli():
    """Test CLI availability"""
    print_header("6. Testing CLI")

    try:
        from app import cli
        print("✅ CLI module available")
        print("\n   To run interactive mode:")
        print("      python -m app.cli query --interactive")
        print("   Or use:")
        print("      make run")
        return True
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  Academic RAG System - System Test")
    print("="*60)

    results = {
        "Imports": test_imports(),
        "Configuration": test_config(),
        "OpenAI Client": test_openai_client(),
        "Vector Store": test_vector_store(),
        "RAG Pipeline": test_rag_pipeline(),
        "CLI": test_cli(),
    }

    # Summary
    print_header("Test Summary")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}  {test_name}")

    print(f"\n   Score: {passed}/{total} tests passed")

    if passed == total:
        print("\n   🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print(f"\n   ⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

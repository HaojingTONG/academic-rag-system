"""
Unit tests for RAG retriever module

Run:
    pytest tests/unit/test_retriever.py -v
    pytest tests/unit/test_retriever.py::TestVectorStore -v
"""

import pytest
from rag.retriever import (
    VectorStore,
    BM25Retriever,
    HybridRetriever,
    QueryAnalyzer,
    RetrievalResult
)


class TestRetrievalResult:
    """Test RetrievalResult data class"""

    def test_creation(self):
        """Test creating a retrieval result"""
        result = RetrievalResult(
            doc_id="test_1",
            content="This is a test document about transformers.",
            metadata={"title": "Test Paper"},
            vector_score=0.95
        )

        assert result.doc_id == "test_1"
        assert "transformers" in result.content
        assert result.vector_score == 0.95
        assert result.metadata["title"] == "Test Paper"

    def test_to_dict(self):
        """Test converting to dictionary"""
        result = RetrievalResult(
            doc_id="test_1",
            content="Test content",
            metadata={},
            vector_score=0.8,
            bm25_score=0.6,
            combined_score=0.74
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["doc_id"] == "test_1"
        assert result_dict["vector_score"] == 0.8
        assert result_dict["bm25_score"] == 0.6


class TestQueryAnalyzer:
    """Test query understanding and expansion"""

    def test_analyze_query(self):
        """Test query analysis"""
        analyzer = QueryAnalyzer()
        intent = analyzer.analyze("What is the transformer architecture?")

        assert intent.intent_type in ['definition', 'general']
        assert isinstance(intent.keywords, list)
        assert len(intent.keywords) > 0
        assert isinstance(intent.expanded_query, str)

    def test_keyword_extraction(self):
        """Test keyword extraction"""
        analyzer = QueryAnalyzer()
        keywords = analyzer._extract_keywords("What is BERT and how does it work?")

        assert "BERT" in keywords
        assert "work" in keywords
        # Stop words should be removed
        assert "is" not in keywords
        assert "and" not in keywords

    def test_query_expansion(self):
        """Test query expansion with synonyms"""
        analyzer = QueryAnalyzer()
        expanded = analyzer._expand_query("transformer architecture")

        # Should contain original query
        assert "transformer" in expanded.lower()
        # Might contain synonyms
        assert len(expanded) >= len("transformer architecture")


class TestBM25Retriever:
    """Test BM25 keyword retrieval"""

    def test_initialization(self):
        """Test BM25 retriever initialization"""
        retriever = BM25Retriever()

        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.documents == []

    def test_fit_and_search(self):
        """Test BM25 fitting and search"""
        documents = [
            "The transformer architecture uses self-attention mechanisms.",
            "BERT is a bidirectional transformer encoder.",
            "GPT is an autoregressive language model."
        ]

        retriever = BM25Retriever()
        retriever.fit(documents)

        # Search
        results = retriever.search("transformer", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        # First result should be most relevant
        assert "transformer" in results[0].content.lower()


@pytest.mark.slow
class TestVectorStore:
    """Test vector store (requires model loading - slow)"""

    @pytest.fixture
    def sample_docs(self):
        """Sample documents for testing"""
        return [
            "The Transformer architecture relies on self-attention.",
            "BERT uses masked language modeling for pretraining.",
            "Vision transformers apply transformers to image patches."
        ]

    @pytest.fixture
    def sample_metadata(self):
        """Sample metadata"""
        return [
            {"title": "Attention Is All You Need", "paper_id": "1"},
            {"title": "BERT", "paper_id": "2"},
            {"title": "ViT", "paper_id": "3"}
        ]

    def test_vector_store_creation(self, tmp_path):
        """Test creating vector store"""
        store = VectorStore(persist_directory=str(tmp_path / "test_db"))

        assert store is not None
        assert hasattr(store, 'collection')
        assert hasattr(store, 'embedding_manager')

    @pytest.mark.skipif(
        True,  # Skip by default to avoid slow tests
        reason="Requires embedding model download"
    )
    def test_add_and_search(self, tmp_path, sample_docs, sample_metadata):
        """Test adding documents and searching"""
        store = VectorStore(persist_directory=str(tmp_path / "test_db"))

        # Add documents
        doc_ids = store.add_documents(sample_docs, sample_metadata)

        assert len(doc_ids) == len(sample_docs)

        # Search
        results = store.search("transformer attention", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(r, RetrievalResult) for r in results)


class TestHybridRetriever:
    """Test hybrid retrieval"""

    def test_weighted_fusion(self):
        """Test weighted score fusion"""
        # Create mock results
        vector_results = [
            RetrievalResult(
                doc_id="doc1",
                content="Content about transformers and attention",
                metadata={},
                vector_score=0.9
            )
        ]

        bm25_results = [
            RetrievalResult(
                doc_id="doc1",
                content="Content about transformers and attention",
                metadata={},
                bm25_score=0.7
            )
        ]

        # Create mock vector store and BM25
        class MockVectorStore:
            def search(self, query, top_k=5):
                return vector_results

        class MockBM25:
            def search(self, query, top_k=5):
                return bm25_results

        # Test hybrid retriever
        retriever = HybridRetriever(
            vector_store=MockVectorStore(),
            bm25_retriever=MockBM25(),
            vector_weight=0.7,
            bm25_weight=0.3
        )

        # Fusion
        fused = retriever._weighted_fusion(vector_results, bm25_results)

        assert len(fused) > 0
        # Combined score should be weighted average
        expected_score = 0.9 * 0.7 + 0.7 * 0.3
        assert abs(fused[0].combined_score - expected_score) < 0.01


# ============================================================================
# Integration Tests (mark as integration)
# ============================================================================

@pytest.mark.integration
class TestRetrievalIntegration:
    """Integration tests for full retrieval pipeline"""

    def test_end_to_end_retrieval(self):
        """Test complete retrieval workflow"""
        # This would test the full pipeline
        # Skipped in unit tests
        pytest.skip("Integration test - run separately")

"""
Unified Retrieval Module for Academic RAG System
=================================================

This module consolidates all retrieval functionality:
- Vector-based semantic search (ChromaDB)
- BM25 keyword search
- Hybrid retrieval (combining vector + BM25)
- Query understanding and expansion
- Adaptive top-k selection

Replaces:
- src/retriever/vector_store.py
- src/retriever/enhanced_vector_store.py
- src/retriever/advanced_retrieval.py (retrieval part)
- src/retriever/enhanced_vector_retrieval.py
"""

import re
import math
import numpy as np
import chromadb
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration
try:
    from configs import config
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    print("⚠️ Config not found, using default values")

# Import embedding manager
try:
    from src.embedding.advanced_embedding import EmbeddingManager, EmbeddingConfig
except ImportError:
    print("⚠️ EmbeddingManager not found, will use fallback")
    EmbeddingManager = None


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RetrievalResult:
    """Unified retrieval result (single source of truth)"""
    doc_id: str
    content: str
    metadata: Dict
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rerank_score: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'doc_id': self.doc_id,
            'content': self.content,
            'metadata': self.metadata,
            'vector_score': self.vector_score,
            'bm25_score': self.bm25_score,
            'combined_score': self.combined_score,
            'rerank_score': self.rerank_score
        }


@dataclass
class QueryIntent:
    """Query understanding result"""
    intent_type: str  # comparison, definition, method, recent_work, general
    confidence: float
    keywords: List[str]
    expanded_query: str


# ============================================================================
# Query Analyzer
# ============================================================================

class QueryAnalyzer:
    """Query understanding and expansion (unified from multiple sources)"""

    def __init__(self):
        # Intent classification patterns
        self.intent_patterns = {
            'comparison': [
                r'\b(compare|comparison|versus|vs|difference|differ)\b',
                r'\b(better|worse|advantage|disadvantage)\b',
                r'\b(which|what.*best|superior|inferior)\b'
            ],
            'definition': [
                r'\b(what is|define|definition|meaning|explain)\b',
                r'\b(how.*work|principle|concept)\b'
            ],
            'method': [
                r'\b(how to|method|approach|technique|algorithm)\b',
                r'\b(implement|implementation|steps|procedure)\b'
            ],
            'recent_work': [
                r'\b(recent|latest|new|current|state-of-the-art)\b',
                r'\b(progress|advance|development|trend)\b'
            ],
            'general': [r'.*']  # Default
        }

        # AI domain synonyms
        self.synonyms = {
            'transformer': ['attention mechanism', 'self-attention', 'multi-head attention'],
            'cnn': ['convolutional neural network', 'convolution', 'conv net'],
            'rnn': ['recurrent neural network', 'LSTM', 'GRU', 'sequence model'],
            'gan': ['generative adversarial network', 'generator', 'discriminator'],
            'bert': ['bidirectional encoder', 'transformer encoder', 'pre-trained model'],
            'gpt': ['generative pre-trained transformer', 'autoregressive model'],
            'attention': ['attention mechanism', 'self-attention', 'cross-attention'],
            'optimization': ['gradient descent', 'backpropagation', 'training'],
            'classification': ['categorization', 'prediction', 'supervised learning']
        }

    def analyze(self, query: str) -> QueryIntent:
        """Analyze query intent and extract keywords"""
        query_lower = query.lower()

        # Intent recognition
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, query_lower))
            intent_scores[intent] = score

        # Determine primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        intent_type = primary_intent[0]
        confidence = min(primary_intent[1] / len(self.intent_patterns[intent_type]), 1.0)

        # Extract keywords
        keywords = self._extract_keywords(query)

        # Expand query
        expanded_query = self._expand_query(query)

        return QueryIntent(
            intent_type=intent_type,
            confidence=confidence,
            keywords=keywords,
            expanded_query=expanded_query
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords

    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms"""
        expanded_terms = []
        query_lower = query.lower()

        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                expanded_terms.extend(synonyms[:2])  # Limit expansion

        if expanded_terms:
            return f"{query} {' '.join(expanded_terms[:3])}"
        return query


# ============================================================================
# BM25 Retriever
# ============================================================================

class BM25Retriever:
    """BM25 sparse retrieval (keyword-based)"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.doc_ids = []
        self.doc_metadatas = []

    def fit(self, documents: List[str], doc_ids: List[str] = None, metadatas: List[Dict] = None):
        """Train BM25 model on documents"""
        self.documents = documents
        self.doc_ids = doc_ids or [f"doc_{i}" for i in range(len(documents))]
        self.doc_metadatas = metadatas or [{} for _ in documents]

        # Tokenize documents
        processed_docs = []
        for doc in documents:
            words = self._tokenize(doc)
            processed_docs.append(words)
            self.doc_len.append(len(words))

        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0

        # Calculate document frequency and IDF
        df = defaultdict(int)
        for doc in processed_docs:
            unique_words = set(doc)
            for word in unique_words:
                df[word] += 1

        # IDF calculation
        num_docs = len(documents)
        for word, freq in df.items():
            self.idf[word] = math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

        # Calculate term frequencies
        self.doc_freqs = []
        for doc in processed_docs:
            freq_dict = defaultdict(int)
            for word in doc:
                freq_dict[word] += 1
            self.doc_freqs.append(freq_dict)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())

    def search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Search using BM25"""
        query_words = self._tokenize(query)
        scores = []

        for doc_idx in range(len(self.documents)):
            score = 0
            doc_freq = self.doc_freqs[doc_idx]
            doc_length = self.doc_len[doc_idx]

            for word in query_words:
                if word in doc_freq and word in self.idf:
                    tf = doc_freq[word]
                    idf = self.idf[word]

                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
                    score += idf * (numerator / denominator)

            scores.append((doc_idx, score))

        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in scores[:top_k]:
            results.append(RetrievalResult(
                doc_id=self.doc_ids[doc_idx],
                content=self.documents[doc_idx],
                metadata=self.doc_metadatas[doc_idx],
                bm25_score=score
            ))

        return results


# ============================================================================
# Vector Store
# ============================================================================

class VectorStore:
    """Unified vector store (ChromaDB + Embeddings)"""

    def __init__(self,
                 persist_directory: str = "vector_db",
                 collection_name: str = "papers_bge-m3_1024_v1",
                 embedding_model: str = None,
                 embedding_config: 'EmbeddingConfig' = None):
        """
        Initialize vector store

        Args:
            persist_directory: Database path
            collection_name: Collection name
            embedding_model: Embedding model name
            embedding_config: Custom embedding configuration
        """
        # Load config
        if USE_CONFIG and embedding_model is None:
            embedding_model = config.embedding.model
        elif embedding_model is None:
            embedding_model = "all-mpnet-base-v2"

        print(f"🚀 Initializing Vector Store...")
        print(f"   - Model: {embedding_model}")
        print(f"   - Collection: {collection_name}")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize embedding manager
        if EmbeddingManager is not None:
            if embedding_config is None:
                from src.embedding.advanced_embedding import EmbeddingConfig
                embedding_config = EmbeddingConfig(
                    model_name=embedding_model,
                    model_type="sentence_transformers",
                    batch_size=32 if not USE_CONFIG else config.embedding.batch_size,
                    normalize_embeddings=True,
                    cache_enabled=True if not USE_CONFIG else config.performance.enable_embedding_cache,
                    cache_dir="data/embedding_cache"
                )

            self.embedding_manager = EmbeddingManager(embedding_config)
            self.embedding_manager.initialize()
            self.model_info = self.embedding_manager.get_model_info()
        else:
            # Fallback to sentence_transformers directly
            from sentence_transformers import SentenceTransformer
            import torch
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.embedding_manager = SentenceTransformer(embedding_model, device=device)
            self.model_info = {
                'model_name': embedding_model,
                'dimension': self.embedding_manager.get_sentence_embedding_dimension(),
                'device': device
            }

        print(f"✅ Vector Store ready")
        print(f"   - Dimension: {self.model_info.get('dimension', 'unknown')}")
        print(f"   - Device: {self.model_info.get('device', 'unknown')}")

    def add_documents(self,
                     documents: List[str],
                     metadatas: List[Dict],
                     doc_ids: List[str] = None) -> List[str]:
        """Add documents to vector store"""
        if not documents or not metadatas:
            print("⚠️ Documents or metadatas empty")
            return []

        print(f"📝 Adding {len(documents)} documents...")

        # Generate IDs if not provided
        if doc_ids is None:
            doc_ids = [f"doc_{i}_{metadatas[i].get('chunk_id', 'unknown')}"
                      for i in range(len(documents))]

        # Generate embeddings
        print("🔢 Generating embeddings...")
        if hasattr(self.embedding_manager, 'encode'):
            # EmbeddingManager or SentenceTransformer
            embeddings = self.embedding_manager.encode(documents)
        else:
            embeddings = self.embedding_manager.encode(documents)

        # Convert to list if numpy array
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        # Batch add to database
        batch_size = 50
        total_added = 0

        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))

            try:
                self.collection.add(
                    embeddings=embeddings[i:end_idx],
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=doc_ids[i:end_idx]
                )

                total_added += (end_idx - i)
                print(f"✅ Added {total_added}/{len(documents)}")

            except Exception as e:
                print(f"❌ Batch {i//batch_size + 1} failed: {e}")
                continue

        print("✅ Documents added successfully")
        return doc_ids

    def search(self,
               query: str,
               top_k: int = 5,
               filter_metadata: Dict = None,
               similarity_threshold: float = None) -> List[RetrievalResult]:
        """Vector similarity search"""
        # Generate query embedding
        if hasattr(self.embedding_manager, 'encode'):
            query_embedding = self.embedding_manager.encode(query)
        else:
            query_embedding = self.embedding_manager.encode([query])

        # Ensure correct format
        if isinstance(query_embedding, np.ndarray):
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.tolist()

        # Build search parameters
        search_params = {
            "query_embeddings": query_embedding,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }

        if filter_metadata:
            search_params["where"] = filter_metadata

        # Search
        try:
            results = self.collection.query(**search_params)

            # Convert to RetrievalResult objects
            retrieval_results = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                similarity_score = 1.0 - distance  # Convert distance to similarity

                # Apply similarity threshold if specified
                if similarity_threshold is not None and similarity_score < similarity_threshold:
                    continue

                retrieval_results.append(RetrievalResult(
                    doc_id=f"result_{i}",
                    content=doc,
                    metadata=metadata,
                    vector_score=similarity_score
                ))

            return retrieval_results

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "embedding_model": self.model_info.get("model_name", "unknown"),
                "embedding_dimension": self.model_info.get("dimension", "unknown"),
                "device": self.model_info.get("device", "unknown")
            }
        except Exception as e:
            return {"error": str(e)}


# ============================================================================
# Hybrid Retriever
# ============================================================================

class HybridRetriever:
    """Hybrid retrieval combining vector search + BM25"""

    def __init__(self,
                 vector_store: VectorStore,
                 bm25_retriever: BM25Retriever = None,
                 vector_weight: float = None,
                 bm25_weight: float = None):
        """
        Initialize hybrid retriever

        Args:
            vector_store: Vector store instance
            bm25_retriever: BM25 retriever instance (optional)
            vector_weight: Weight for vector scores
            bm25_weight: Weight for BM25 scores
        """
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

        # Load weights from config
        if USE_CONFIG:
            self.vector_weight = vector_weight or config.retrieval.vector_weight
            self.bm25_weight = bm25_weight or config.retrieval.bm25_weight
        else:
            self.vector_weight = vector_weight or 0.7
            self.bm25_weight = bm25_weight or 0.3

        # Query analyzer
        self.query_analyzer = QueryAnalyzer()

        print(f"🔀 Hybrid Retriever initialized")
        print(f"   - Vector weight: {self.vector_weight}")
        print(f"   - BM25 weight: {self.bm25_weight}")

    def retrieve(self,
                 query: str,
                 top_k: int = 5,
                 use_query_expansion: bool = True,
                 fusion_method: str = "weighted") -> List[RetrievalResult]:
        """
        Hybrid retrieval

        Args:
            query: Query string
            top_k: Number of results
            use_query_expansion: Whether to expand query
            fusion_method: How to combine scores (weighted, rrf)

        Returns:
            List of retrieval results
        """
        # Query analysis and expansion
        if use_query_expansion:
            query_intent = self.query_analyzer.analyze(query)
            search_query = query_intent.expanded_query
        else:
            search_query = query

        # Vector search
        vector_results = self.vector_store.search(search_query, top_k=top_k * 2)

        # BM25 search (if available)
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.search(search_query, top_k=top_k * 2)
        else:
            bm25_results = []

        # Fusion
        if fusion_method == "weighted":
            combined_results = self._weighted_fusion(vector_results, bm25_results)
        elif fusion_method == "rrf":
            combined_results = self._reciprocal_rank_fusion(vector_results, bm25_results)
        else:
            combined_results = vector_results  # Fallback to vector only

        # Return top-k
        return combined_results[:top_k]

    def _weighted_fusion(self,
                        vector_results: List[RetrievalResult],
                        bm25_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Weighted score fusion"""
        # Index results by content for matching
        result_map = {}

        # Add vector results
        for result in vector_results:
            key = result.content[:100]  # Use first 100 chars as key
            result_map[key] = result
            result.combined_score = result.vector_score * self.vector_weight

        # Add BM25 results
        for result in bm25_results:
            key = result.content[:100]
            if key in result_map:
                # Update existing result
                result_map[key].bm25_score = result.bm25_score
                result_map[key].combined_score += result.bm25_score * self.bm25_weight
            else:
                # New result from BM25 only
                result.combined_score = result.bm25_score * self.bm25_weight
                result_map[key] = result

        # Sort by combined score
        combined_results = sorted(result_map.values(),
                                 key=lambda x: x.combined_score,
                                 reverse=True)

        return combined_results

    def _reciprocal_rank_fusion(self,
                                vector_results: List[RetrievalResult],
                                bm25_results: List[RetrievalResult],
                                k: int = 60) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion (RRF)"""
        rrf_scores = defaultdict(float)
        result_map = {}

        # Add vector results
        for rank, result in enumerate(vector_results, 1):
            key = result.content[:100]
            rrf_scores[key] += 1.0 / (k + rank)
            result_map[key] = result

        # Add BM25 results
        for rank, result in enumerate(bm25_results, 1):
            key = result.content[:100]
            rrf_scores[key] += 1.0 / (k + rank)
            if key not in result_map:
                result_map[key] = result

        # Update combined scores
        for key, score in rrf_scores.items():
            result_map[key].combined_score = score

        # Sort by RRF score
        combined_results = sorted(result_map.values(),
                                 key=lambda x: x.combined_score,
                                 reverse=True)

        return combined_results


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'RetrievalResult',
    'QueryIntent',
    'QueryAnalyzer',
    'BM25Retriever',
    'VectorStore',
    'HybridRetriever'
]

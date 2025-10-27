"""
RAG Pipeline - End-to-End Orchestration
========================================

This module provides the main RAG pipeline that orchestrates:
1. Retrieval (vector + BM25 hybrid)
2. Reranking (cross-encoder or feature-based)
3. Prompt composition
4. Generation (LLM)
5. Quality checks (optional)

Usage:
    from rag.pipeline import RAGPipeline, RAGConfig

    # Initialize pipeline
    pipeline = RAGPipeline()

    # Query
    result = pipeline.query("What is the transformer architecture?")

    # Access answer and sources
    print(result['answer'])
    print(result['sources'])
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sys

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config
try:
    from configs import config
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False
    print("⚠️ Config not available, using defaults")

# Import retrieval components
try:
    from rag.retriever import VectorStore, HybridRetriever, BM25Retriever, RetrievalResult
    from rag.ranker import CompositeRanker
except ImportError as e:
    print(f"⚠️ Failed to import rag modules: {e}")
    VectorStore = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RAGConfig:
    """RAG Pipeline Configuration"""
    # Retrieval
    top_k: int = 5
    use_hybrid_retrieval: bool = True
    vector_weight: float = 0.7
    bm25_weight: float = 0.3

    # Reranking
    enable_reranking: bool = True
    enable_diversification: bool = True
    enable_filtering: bool = True

    # Generation
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.1
    max_tokens: int = 2000

    # Quality
    include_sources: bool = True
    enable_fact_check: bool = False
    enable_citations: bool = True

    # Performance
    max_context_length: int = 4000
    timeout: int = 60

    @classmethod
    def from_global_config(cls) -> 'RAGConfig':
        """Create from global config"""
        if not USE_CONFIG:
            return cls()

        return cls(
            top_k=config.retrieval.top_k,
            use_hybrid_retrieval=config.retrieval.enable_bm25,
            vector_weight=config.retrieval.vector_weight,
            bm25_weight=config.retrieval.bm25_weight,
            enable_reranking=config.retrieval.enable_rerank,
            enable_diversification=config.retrieval.enable_mmr,
            enable_filtering=True,
            llm_model=config.generation.model,
            llm_temperature=config.generation.temperature,
            max_tokens=config.generation.max_tokens,
            include_sources=config.generation.include_sources,
            max_context_length=config.generation.max_context_length,
            timeout=config.generation.timeout
        )


# ============================================================================
# RAG Pipeline
# ============================================================================

class RAGPipeline:
    """
    Main RAG Pipeline

    Orchestrates the full RAG workflow:
    Query → Retrieve → Rank → Prompt → Generate → Response
    """

    def __init__(self, rag_config: Optional[RAGConfig] = None):
        """
        Initialize RAG pipeline

        Args:
            rag_config: RAG configuration (if None, loads from global config)
        """
        print("🚀 Initializing RAG Pipeline...")

        # Load config
        if rag_config is None:
            if USE_CONFIG:
                self.config = RAGConfig.from_global_config()
            else:
                self.config = RAGConfig()
        else:
            self.config = rag_config

        # Initialize components
        self.vector_store = None
        self.retriever = None
        self.ranker = None
        self.llm_client = None

        print("✅ RAG Pipeline initialized")

    def initialize(self,
                   vector_store: VectorStore = None,
                   bm25_retriever: BM25Retriever = None,
                   llm_client = None):
        """
        Initialize pipeline components

        Args:
            vector_store: Pre-initialized vector store
            bm25_retriever: Pre-initialized BM25 retriever
            llm_client: Pre-initialized LLM client
        """
        print("🔧 Initializing components...")

        # Vector store
        if vector_store is not None:
            self.vector_store = vector_store
        elif VectorStore is not None:
            self.vector_store = VectorStore()
        else:
            raise RuntimeError("VectorStore not available")

        # Retriever
        if self.config.use_hybrid_retrieval and bm25_retriever is not None:
            self.retriever = HybridRetriever(
                vector_store=self.vector_store,
                bm25_retriever=bm25_retriever,
                vector_weight=self.config.vector_weight,
                bm25_weight=self.config.bm25_weight
            )
        else:
            # Fall back to vector-only retrieval
            self.retriever = self.vector_store
            print("⚠️ Using vector-only retrieval (BM25 not available)")

        # Ranker
        self.ranker = CompositeRanker(
            enable_reranking=self.config.enable_reranking,
            enable_filtering=self.config.enable_filtering,
            enable_diversification=self.config.enable_diversification,
            enable_deduplication=True
        )

        # LLM Client
        if llm_client is not None:
            self.llm_client = llm_client
        else:
            # Try to import and initialize LLM client
            try:
                from src.generator.llm_client import OllamaClient
                self.llm_client = OllamaClient(
                    model=self.config.llm_model,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.max_tokens
                )
            except Exception as e:
                print(f"⚠️ Could not initialize LLM client: {e}")
                self.llm_client = None

        print("✅ Components initialized")

    def query(self,
              question: str,
              top_k: Optional[int] = None,
              custom_prompt: Optional[str] = None,
              return_metadata: bool = True) -> Dict:
        """
        Execute RAG query

        Args:
            question: User question
            top_k: Number of documents to retrieve (overrides config)
            custom_prompt: Custom prompt template
            return_metadata: Whether to include metadata in response

        Returns:
            Dictionary with answer, sources, and metadata
        """
        start_time = time.time()

        # Use configured top_k if not specified
        if top_k is None:
            top_k = self.config.top_k

        print(f"\n{'='*60}")
        print(f"📝 Query: {question}")
        print(f"{'='*60}\n")

        try:
            # Step 1: Retrieval
            print("🔍 [1/4] Retrieving documents...")
            retrieval_start = time.time()

            if hasattr(self.retriever, 'retrieve'):
                # Hybrid retriever
                results = self.retriever.retrieve(
                    query=question,
                    top_k=top_k * 2,  # Retrieve more for reranking
                    use_query_expansion=True
                )
            else:
                # Vector-only retriever
                results = self.retriever.search(
                    query=question,
                    top_k=top_k * 2
                )

            retrieval_time = time.time() - retrieval_start
            print(f"   ✓ Retrieved {len(results)} documents ({retrieval_time:.2f}s)")

            # Step 2: Ranking
            print("\n📊 [2/4] Ranking and filtering...")
            ranking_start = time.time()

            results = self.ranker.process(
                query=question,
                results=results,
                top_k=top_k
            )

            ranking_time = time.time() - ranking_start
            print(f"   ✓ Ranked to {len(results)} documents ({ranking_time:.2f}s)")

            # Step 3: Prompt Composition
            print("\n✍️  [3/4] Composing prompt...")
            prompt = self._compose_prompt(
                question=question,
                results=results,
                custom_template=custom_prompt
            )

            # Step 4: Generation
            print("\n🤖 [4/4] Generating answer...")
            generation_start = time.time()

            if self.llm_client is not None:
                answer = self._generate_answer(prompt)
            else:
                answer = "[LLM client not available - showing context only]"

            generation_time = time.time() - generation_start
            print(f"   ✓ Answer generated ({generation_time:.2f}s)")

            # Prepare response
            total_time = time.time() - start_time

            response = {
                'answer': answer,
                'sources': self._format_sources(results),
                'num_sources': len(results),
                'success': True
            }

            if return_metadata:
                response['metadata'] = {
                    'retrieval_time': retrieval_time,
                    'ranking_time': ranking_time,
                    'generation_time': generation_time,
                    'total_time': total_time,
                    'top_k': top_k,
                    'config': asdict(self.config)
                }

            print(f"\n{'='*60}")
            print(f"✅ Query completed in {total_time:.2f}s")
            print(f"{'='*60}\n")

            return response

        except Exception as e:
            print(f"\n❌ Query failed: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'sources': [],
                'num_sources': 0,
                'success': False,
                'error': str(e)
            }

    def _compose_prompt(self,
                       question: str,
                       results: List[RetrievalResult],
                       custom_template: Optional[str] = None) -> str:
        """Compose prompt from question and context"""
        # Format context from results
        context_parts = []
        for idx, result in enumerate(results, 1):
            source_info = result.metadata.get('title', 'Unknown')
            context_parts.append(f"[{idx}] {result.content}\n(Source: {source_info})")

        context = "\n\n".join(context_parts)

        # Use custom template or default
        if custom_template:
            prompt = custom_template.format(context=context, question=question)
        else:
            prompt = self._default_prompt_template(question, context)

        return prompt

    def _default_prompt_template(self, question: str, context: str) -> str:
        """Default prompt template"""
        return f"""You are an AI assistant helping researchers understand academic papers.

Context (relevant excerpts from research papers):
{context}

Question: {question}

Instructions:
- Answer based on the provided context
- Be precise and technical when appropriate
- Cite sources using [1], [2], etc. when referencing specific information
- If the context doesn't fully answer the question, acknowledge limitations
- Use clear, structured explanations

Answer:"""

    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using LLM"""
        try:
            # Different LLM client interfaces
            if hasattr(self.llm_client, 'generate'):
                response = self.llm_client.generate(prompt)
                if isinstance(response, dict):
                    return response.get('response', str(response))
                return str(response)
            elif hasattr(self.llm_client, 'query'):
                return self.llm_client.query(prompt)
            elif callable(self.llm_client):
                return self.llm_client(prompt)
            else:
                return "[LLM client interface not recognized]"

        except Exception as e:
            return f"[Generation error: {e}]"

    def _format_sources(self, results: List[RetrievalResult]) -> List[Dict]:
        """Format sources for response"""
        sources = []

        for idx, result in enumerate(results, 1):
            source = {
                'index': idx,
                'content': result.content[:200] + "..." if len(result.content) > 200 else result.content,
                'metadata': result.metadata,
                'score': max(result.rerank_score, result.combined_score, result.vector_score)
            }

            # Add paper info if available
            if 'title' in result.metadata:
                source['title'] = result.metadata['title']
            if 'paper_id' in result.metadata:
                source['paper_id'] = result.metadata['paper_id']

            sources.append(source)

        return sources

    def batch_query(self, questions: List[str], **kwargs) -> List[Dict]:
        """
        Process multiple queries

        Args:
            questions: List of questions
            **kwargs: Additional arguments for query()

        Returns:
            List of results
        """
        results = []
        for question in questions:
            result = self.query(question, **kwargs)
            results.append(result)

        return results

    def get_status(self) -> Dict:
        """Get pipeline status"""
        status = {
            'initialized': all([
                self.vector_store is not None,
                self.retriever is not None,
                self.ranker is not None
            ]),
            'vector_store': self.vector_store is not None,
            'retriever': self.retriever is not None,
            'ranker': self.ranker is not None,
            'llm_client': self.llm_client is not None,
            'config': asdict(self.config)
        }

        if self.vector_store:
            try:
                stats = self.vector_store.get_stats()
                status['vector_store_stats'] = stats
            except:
                pass

        return status


# ============================================================================
# Module Exports
# ============================================================================

__all__ = ['RAGPipeline', 'RAGConfig']

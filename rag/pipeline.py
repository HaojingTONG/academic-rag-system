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

# Import new modular components
try:
    from rag.composer import PromptComposer, PromptConfig, compose_empty_response
    from rag.generator import RAGGenerator, GenerationConfig, ProviderType
    from rag.citation_utils import sanitize_sources, add_citation_warnings, get_cited_numbers
except ImportError as e:
    print(f"⚠️ Failed to import new RAG modules: {e}")
    PromptComposer = None
    RAGGenerator = None


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

        # Initialize new modular components
        self.composer = None
        self.generator = None

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
            # Pass collection_name from config
            collection_name = config.vector_store.collection_name if USE_CONFIG else "papers_bge-m3_1024_v1"
            self.vector_store = VectorStore(collection_name=collection_name)
        else:
            raise RuntimeError("VectorStore not available")

        # BM25 Retriever - auto-load from disk if available
        if bm25_retriever is None and self.config.use_hybrid_retrieval:
            bm25_path = Path("data/bm25_index.pkl")
            if bm25_path.exists():
                try:
                    import pickle
                    print(f"📦 Loading BM25 index from {bm25_path}...")
                    with open(bm25_path, 'rb') as f:
                        bm25_retriever = pickle.load(f)
                    print(f"   ✓ BM25 index loaded ({len(bm25_retriever.documents)} docs)")
                except Exception as e:
                    print(f"⚠️ Failed to load BM25 index: {e}")
                    bm25_retriever = None

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
            # Try to import and initialize LLM client from config
            try:
                from src.generator.llm_client import create_llm_manager_from_config
                self.llm_client = create_llm_manager_from_config()
                print(f"✅ LLM Manager initialized: {self.llm_client.backend}")
            except Exception as e:
                print(f"⚠️ Could not initialize LLM client: {e}")
                self.llm_client = None

        # Prompt Composer
        if PromptComposer:
            prompt_config = PromptConfig(
                max_context_length=self.config.max_context_length,
                include_metadata=True,
                context_style="detailed"
            )
            self.composer = PromptComposer(prompt_config)
            print(f"✅ Prompt Composer initialized")
        else:
            print(f"⚠️ PromptComposer not available")

        # RAG Generator with retry and fallback
        if RAGGenerator:
            gen_config = GenerationConfig(
                primary_provider=ProviderType.OLLAMA,
                max_retries=3,
                max_tokens=self.config.max_tokens,
                temperature=self.config.llm_temperature
            )
            self.generator = RAGGenerator(gen_config)
            print(f"✅ RAG Generator initialized (retry + multi-provider fallback)")
        else:
            print(f"⚠️ RAGGenerator not available")

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
            # Step 0.5: Query Analysis and Rewriting
            print("📋 [0.5/5] Analyzing query...")
            try:
                from rag.query_rewriter import QueryRewriter
                rewriter = QueryRewriter()
                query_intent = rewriter.detect_intent(question)

                print(f"   ✓ Detected intent: {query_intent.intent_type}")
                print(f"   ✓ Is general query: {query_intent.is_general}")
                print(f"   ✓ Key concepts: {', '.join(query_intent.key_concepts) if query_intent.key_concepts else 'None'}")

                # Expand query for better retrieval
                expanded_queries = rewriter.expand_query(question, max_queries=3)
                print(f"   ✓ Expanded to {len(expanded_queries)} queries")
                for i, eq in enumerate(expanded_queries[:3], 1):
                    print(f"      {i}. {eq[:80]}...")

            except Exception as e:
                print(f"   ⚠️  Query analysis failed: {e}")
                query_intent = None
                expanded_queries = [question]

            # Step 1: Multi-Query Retrieval
            print("\n🔍 [1/5] Retrieving documents...")
            retrieval_start = time.time()

            # Retrieve with multiple query variations and merge results
            all_results = []
            for query_variant in expanded_queries:
                if hasattr(self.retriever, 'retrieve'):
                    # Hybrid retriever
                    variant_results = self.retriever.retrieve(
                        query=query_variant,
                        top_k=top_k * 2,  # Retrieve more for reranking
                        use_query_expansion=False  # Already expanded
                    )
                else:
                    # Vector-only retriever
                    variant_results = self.retriever.search(
                        query=query_variant,
                        top_k=top_k * 2
                    )
                all_results.extend(variant_results)

            # Deduplicate by content hash
            seen_content = set()
            results = []
            for r in all_results:
                content_hash = hash(r.content[:200])  # Hash first 200 chars
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    results.append(r)

            retrieval_time = time.time() - retrieval_start
            print(f"   ✓ Retrieved {len(results)} unique documents ({retrieval_time:.2f}s)")

            # Step 2: Ranking
            print("\n📊 [2/5] Ranking and filtering...")
            ranking_start = time.time()

            results = self.ranker.process(
                query=question,
                results=results,
                top_k=top_k * 2  # Get more for relevance filtering
            )

            ranking_time = time.time() - ranking_start
            print(f"   ✓ Ranked to {len(results)} documents ({ranking_time:.2f}s)")

            # Step 2.5: Relevance Filtering
            print("\n🎯 [2.5/5] Filtering by relevance...")
            try:
                from rag.relevance_filter import RelevanceFilter, check_retrieval_quality

                relevance_filter = RelevanceFilter()

                # Check retrieval quality
                is_general = query_intent.is_general if query_intent else False
                quality_check = check_retrieval_quality(results, is_general)

                print(f"   ✓ Average score: {quality_check['avg_score']:.3f}")
                print(f"   ✓ Above threshold: {quality_check['num_above_threshold']}/{quality_check['total_results']}")
                print(f"   ✓ Quality acceptable: {quality_check['is_acceptable']}")

                # Filter results
                filtered_results = relevance_filter.filter_results(
                    results, question, is_general
                )

                print(f"   ✓ Kept {len(filtered_results)} relevant documents")

                # Check if we should use fallback (general knowledge)
                use_fallback = (
                    is_general and
                    not quality_check['is_acceptable'] and
                    quality_check.get('avg_score', 0) < 0.30
                )

                if use_fallback:
                    print(f"   ⚠️  Low quality retrieval for general question - using fallback")

                results = filtered_results

            except Exception as e:
                print(f"   ⚠️  Relevance filtering failed: {e}")
                use_fallback = False
                results = results[:top_k]  # Just truncate to top_k

            # Check for empty context
            if len(results) == 0:
                print("\n⚠️  WARNING: No relevant documents found")

                # Use general knowledge fallback for general questions
                if query_intent and query_intent.is_general and self.composer:
                    print("   → Using general knowledge fallback")
                    prompt = PromptTemplate.general_knowledge_fallback_template(question)

                    # Generate using fallback template
                    if self.generator:
                        gen_result = self.generator.generate(prompt)
                        empty_answer = gen_result.text
                    else:
                        empty_answer = self._empty_context_response(question)
                else:
                    # Use standard empty response
                    if self.composer:
                        empty_answer = self.composer.compose_empty_context_response(question)
                    else:
                        empty_answer = self._empty_context_response(question)

                return {
                    'answer': empty_answer,
                    'sources': [],
                    'num_sources': 0,
                    'success': True,
                    'warning': 'no_context_found',
                    'used_fallback': query_intent and query_intent.is_general
                }

            # Step 3: Prompt Composition
            print("\n✍️  [3/5] Composing prompt...")

            # Select template based on query intent and retrieval quality
            template_style = "afel"  # Default to A.F.E.L. to prevent paper dumps

            if query_intent:
                # Map intent to template style
                intent_to_template = {
                    'definition': 'afel',  # Use A.F.E.L. for definitions
                    'comparison': 'comparative',
                    'method': 'detailed',
                    'recent_work': 'detailed',
                    'general': 'afel'  # Use A.F.E.L. for general questions
                }
                template_style = intent_to_template.get(query_intent.intent_type, 'afel')
                print(f"   ✓ Query intent: {query_intent.intent_type} → using '{template_style}' template")

                # Use fallback if low quality retrieval for general questions
                if use_fallback:
                    template_style = 'general_fallback'
                    print(f"   ✓ Using general knowledge fallback (low quality retrieval)")

            # Use new composer if available
            if self.composer:
                prompt = self.composer.compose(
                    question=question,
                    results=results,
                    template_style=template_style,
                    custom_template=custom_prompt
                )
            else:
                prompt = self._compose_prompt(
                    question=question,
                    results=results,
                    custom_template=custom_prompt
                )

            # Step 4: Generation with retry and fallback
            print("\n🤖 [4/4] Generating answer...")
            generation_start = time.time()

            # Use new generator with retry if available
            if self.generator:
                gen_result = self.generator.generate(
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.llm_temperature
                )
                answer = gen_result.text
                print(f"   ✓ Generated with {gen_result.provider.value} ({gen_result.attempts} attempts)")
            elif self.llm_client is not None:
                answer = self._generate_answer(prompt, results)
            else:
                answer = "[LLM client not available - showing context only]"

            generation_time = time.time() - generation_start
            print(f"   ✓ Answer generated ({generation_time:.2f}s)")

            # Step 4.5: Answer Quality Enhancement
            print("\n✨ [4.5/5] Enhancing answer quality...")
            formatted_sources = self._format_sources(results)

            try:
                from rag.answer_quality import AnswerEnhancer, FaithfulnessChecker

                # Add summary if missing
                enhancer = AnswerEnhancer()
                answer = enhancer.add_summary(answer, question, formatted_sources)

                # Check faithfulness with keyword validation
                checker = FaithfulnessChecker()
                context_text = '\n'.join([r.content for r in results])
                faithfulness = checker.check_faithfulness(
                    answer=answer,
                    context=context_text,
                    question=question  # Pass question for concept detection
                )

                print(f"   ✓ Faithfulness score: {faithfulness.score:.2f}")
                print(f"   ✓ Citation coverage: {faithfulness.citation_coverage:.1%}")
                print(f"   ✓ Keyword coverage: {faithfulness.keyword_coverage:.1%}")
                print(f"   ✓ Verbatim copying: {faithfulness.verbatim_copying:.1%}")

                if faithfulness.issues:
                    for issue in faithfulness.issues:
                        print(f"   ⚠️  {issue}")

            except Exception as e:
                print(f"   ⚠️  Enhancement skipped: {e}")

            # Step 5: Citation Sanitization
            print("\n🔍 [5/5] Sanitizing citations...")
            # Add warnings for missing/invalid citations
            if add_citation_warnings:
                answer = add_citation_warnings(answer, len(results))

            # Sanitize sources - only keep cited ones
            if sanitize_sources and get_cited_numbers(answer):
                cited_nums = get_cited_numbers(answer)
                print(f"   ✓ Found {len(cited_nums)} unique citations")
                sanitized_sources = sanitize_sources(answer, formatted_sources)
                print(f"   ✓ Kept {len(sanitized_sources)}/{len(formatted_sources)} cited sources")
            else:
                sanitized_sources = formatted_sources

            # Prepare response
            total_time = time.time() - start_time

            response = {
                'answer': answer,
                'sources': sanitized_sources,
                'num_sources': len(sanitized_sources),
                'all_retrieved': len(results),
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

    def _empty_context_response(self, question: str) -> str:
        """Generate safe response when no context is found"""
        return f"""I apologize, but I couldn't find relevant information in the academic paper database to answer your question: "{question}"

This could mean:
- The question is outside the scope of the indexed papers
- The query terms don't match the available content
- The papers relevant to this topic haven't been indexed yet

**Suggestions:**
- Try rephrasing your question with different keywords
- Ask about topics more directly covered in machine learning/AI research papers
- Check if the question is within the domain of computer science research

**Note:** This response is based on a search of the indexed academic papers. I cannot provide information beyond what's in the database."""

    def _generate_answer(self, prompt: str, results: List[RetrievalResult]) -> str:
        """
        Generate answer using LLM with citation validation

        Args:
            prompt: The formatted prompt with context
            results: The retrieval results (for citation validation)

        Returns:
            Generated answer text
        """
        try:
            # LLMManager interface (returns LLMResponse)
            if hasattr(self.llm_client, 'generate_answer'):
                response = self.llm_client.generate_answer(
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.llm_temperature
                )
                # LLMResponse has a .text attribute
                if hasattr(response, 'text'):
                    answer = response.text
                else:
                    answer = str(response)

                # Validate citations exist in response
                return self._validate_citations(answer, results)

            # OllamaClient or similar (returns dict)
            elif hasattr(self.llm_client, 'generate'):
                response = self.llm_client.generate(prompt)
                if isinstance(response, dict):
                    answer = response.get('response', str(response))
                else:
                    answer = str(response)
                return self._validate_citations(answer, results)

            # Direct query interface
            elif hasattr(self.llm_client, 'query'):
                answer = self.llm_client.query(prompt)
                return self._validate_citations(answer, results)

            # Callable interface
            elif callable(self.llm_client):
                answer = self.llm_client(prompt)
                return self._validate_citations(answer, results)
            else:
                return "[LLM client interface not recognized]"

        except Exception as e:
            return f"[Generation error: {e}]"

    def _validate_citations(self, answer: str, results: List[RetrievalResult]) -> str:
        """
        Validate that citations in answer correspond to actual sources

        Args:
            answer: Generated answer text
            results: Retrieved documents

        Returns:
            Validated answer (with warning if citations are invalid)
        """
        import re

        # Find all citation markers like [1], [2], etc.
        citations = re.findall(r'\[(\d+)\]', answer)

        if not citations:
            # No citations found - add warning if we have sources
            if len(results) > 0:
                answer += "\n\n**Note:** This answer should cite specific sources. Please verify information independently."
        else:
            # Validate citation numbers
            max_citation = max([int(c) for c in citations])
            if max_citation > len(results):
                answer += f"\n\n**Warning:** Some citations ([1]-[{len(results)}] available) may be invalid."

        return answer

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

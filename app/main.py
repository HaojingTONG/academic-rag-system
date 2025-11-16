"""
FastAPI Web API for Academic RAG System
========================================

Provides REST API endpoints for RAG queries.

Endpoints:
    POST /query - Query the RAG system
    GET /health - Health check
    GET /stats - System statistics
    GET /docs - API documentation (auto-generated)

Usage:
    # Start server
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

    # Or use Makefile
    make serve
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import time
import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import RAG components
try:
    from rag import RAGPipeline, RAGConfig
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ RAG module not available: {e}")
    RAG_AVAILABLE = False

# Import config
try:
    from configs import config
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False

# ============================================================================
# Pydantic Models
# ============================================================================

class QueryRequest(BaseModel):
    """Query request model"""
    question: str = Field(..., min_length=1, description="Question to ask")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of documents to retrieve")
    enable_reranking: Optional[bool] = Field(True, description="Enable reranking")
    enable_diversification: Optional[bool] = Field(True, description="Enable result diversification")
    include_metadata: Optional[bool] = Field(True, description="Include metadata in response")

    class Config:
        schema_extra = {
            "example": {
                "question": "What is the transformer architecture?",
                "top_k": 5,
                "enable_reranking": True,
                "enable_diversification": True,
                "include_metadata": True
            }
        }


class Source(BaseModel):
    """Source document model"""
    index: int
    content: str
    title: Optional[str] = None
    score: float
    metadata: Dict


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source documents")
    num_sources: int = Field(..., description="Number of sources")
    success: bool = Field(..., description="Whether query succeeded")
    metadata: Optional[Dict] = Field(None, description="Performance metadata")

    class Config:
        schema_extra = {
            "example": {
                "answer": "The Transformer is a neural network architecture...",
                "sources": [
                    {
                        "index": 1,
                        "content": "The Transformer model architecture...",
                        "title": "Attention Is All You Need",
                        "score": 0.95,
                        "metadata": {"paper_id": "1706.03762"}
                    }
                ],
                "num_sources": 5,
                "success": True,
                "metadata": {
                    "retrieval_time": 0.15,
                    "ranking_time": 0.08,
                    "generation_time": 1.2,
                    "total_time": 1.5
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    rag_available: bool
    config_available: bool
    timestamp: float


class StatsResponse(BaseModel):
    """Statistics response"""
    num_documents: int
    embedding_model: str
    embedding_dimension: int
    device: str
    config: Dict


class PaperMetadata(BaseModel):
    """Paper metadata model"""
    paper_id: str
    title: str
    authors: Optional[List[str]] = []
    year: Optional[int] = None
    venue: Optional[str] = None
    arxiv_id: Optional[str] = None
    num_chunks: Optional[int] = 0
    file_path: Optional[str] = None


class ConfigUpdateRequest(BaseModel):
    """Configuration update request"""
    top_k: Optional[int] = Field(None, ge=1, le=20)
    vector_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    bm25_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    enable_reranking: Optional[bool] = None
    enable_diversification: Optional[bool] = None
    llm_temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=100, le=4000)


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Academic RAG System API",
    description="Question answering over academic papers using RAG",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not USE_CONFIG else config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance (initialized on startup)
rag_pipeline: Optional[RAGPipeline] = None


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline

    print("🚀 Starting Academic RAG System API...")

    if not RAG_AVAILABLE:
        print("⚠️ RAG module not available - API will have limited functionality")
        return

    try:
        # Initialize RAG pipeline
        rag_config = RAGConfig.from_global_config() if USE_CONFIG else RAGConfig()
        rag_pipeline = RAGPipeline(rag_config)

        # Initialize components (this might take a few seconds)
        print("🔧 Initializing RAG components...")
        rag_pipeline.initialize()

        print("✅ RAG System API ready!")

    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        rag_pipeline = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down Academic RAG System API...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Academic RAG System API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "query": "POST /query"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if rag_pipeline is not None else "degraded",
        version="2.0.0",
        rag_available=RAG_AVAILABLE and rag_pipeline is not None,
        config_available=USE_CONFIG,
        timestamp=time.time()
    )


@app.get("/health/rag", tags=["System"])
async def health_rag():
    """
    RAG smoke test endpoint

    Runs a test query to verify end-to-end RAG functionality.
    Returns retrieval stats, kept results, and citation info.
    """
    if rag_pipeline is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "RAG system not initialized",
                "retrieved_n": 0,
                "kept_n": 0,
                "citations": []
            }
        )

    try:
        # Run smoke test query
        test_query = "What is Transformer Architecture?"
        start_time = time.time()

        result = rag_pipeline.query(
            question=test_query,
            top_k=5,
            return_metadata=True
        )

        elapsed = time.time() - start_time

        # Extract citations from answer
        import re
        citations = re.findall(r'\[(\d+)\]', result.get('answer', ''))

        # Build response
        response = {
            "status": "healthy" if result.get('success') else "degraded",
            "test_query": test_query,
            "retrieved_n": result.get('metadata', {}).get('retrieved_count', result.get('num_sources', 0)),
            "kept_n": result.get('num_sources', 0),
            "citations": [int(c) for c in citations] if citations else [],
            "has_answer": len(result.get('answer', '')) > 0,
            "latency_ms": round(elapsed * 1000, 2),
            "sources_preview": [
                {
                    "index": s.get('index'),
                    "title": s.get('title', s.get('metadata', {}).get('title', 'Unknown'))[:60],
                    "score": round(s.get('score', 0), 4)
                }
                for s in result.get('sources', [])[:3]
            ],
            "timestamp": time.time()
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "retrieved_n": 0,
                "kept_n": 0,
                "citations": [],
                "timestamp": time.time()
            }
        )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get system statistics"""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )

    try:
        status = rag_pipeline.get_status()
        vector_stats = status.get('vector_store_stats', {})

        return StatsResponse(
            num_documents=vector_stats.get('total_documents', 0),
            embedding_model=vector_stats.get('embedding_model', 'unknown'),
            embedding_dimension=vector_stats.get('embedding_dimension', 0),
            device=vector_stats.get('device', 'unknown'),
            config=status.get('config', {})
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest):
    """
    Query the RAG system

    Send a question and get an answer with sources.

    **Parameters:**
    - **question**: The question to ask (required)
    - **top_k**: Number of source documents to retrieve (1-20, default: 5)
    - **enable_reranking**: Whether to rerank results (default: true)
    - **enable_diversification**: Whether to diversify results (default: true)
    - **include_metadata**: Include performance metadata (default: true)

    **Returns:**
    - **answer**: Generated answer
    - **sources**: List of source documents with scores
    - **num_sources**: Number of sources used
    - **success**: Whether query succeeded
    - **metadata**: Performance metrics (if include_metadata=true)
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please check /health endpoint."
        )

    try:
        # Execute query
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            return_metadata=request.include_metadata
        )

        # Format sources
        sources = [
            Source(
                index=src['index'],
                content=src['content'],
                title=src.get('title'),
                score=src['score'],
                metadata=src['metadata']
            )
            for src in result.get('sources', [])
        ]

        # Build response
        response = QueryResponse(
            answer=result.get('answer', ''),
            sources=sources,
            num_sources=result.get('num_sources', 0),
            success=result.get('success', False),
            metadata=result.get('metadata') if request.include_metadata else None
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@app.get("/models", tags=["System"])
async def list_models():
    """List available models"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        status = rag_pipeline.get_status()
        return {
            "embedding_model": status.get('vector_store_stats', {}).get('embedding_model', 'unknown'),
            "llm_model": status.get('config', {}).get('llm_model', 'unknown'),
            "device": status.get('vector_store_stats', {}).get('device', 'unknown')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Paper Management Endpoints
# ============================================================================

@app.get("/papers", tags=["Papers"])
async def list_papers(
    limit: int = Query(100, ge=1, le=1000, description="Max number of papers to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    search: Optional[str] = Query(None, description="Search in title/authors")
):
    """
    List all indexed papers with metadata

    Returns papers from papers_info.json with chunk counts from vector DB
    """
    try:
        # Load papers info
        papers_file = Path("data/papers_info.json")
        if not papers_file.exists():
            return JSONResponse(content={
                "papers": [],
                "total": 0,
                "limit": limit,
                "offset": offset
            })

        with open(papers_file, 'r', encoding='utf-8') as f:
            papers_data = json.load(f)

        # Convert to list if it's a dict
        if isinstance(papers_data, dict):
            papers_list = list(papers_data.values())
        else:
            papers_list = papers_data

        # Search filter
        if search:
            search_lower = search.lower()
            papers_list = [
                p for p in papers_list
                if search_lower in p.get('title', '').lower()
                or search_lower in ' '.join(p.get('authors', [])).lower()
            ]

        # Sort by year (descending) then title
        papers_list.sort(
            key=lambda p: (-(p.get('year') or 0), p.get('title', ''))
        )

        total = len(papers_list)
        papers_page = papers_list[offset:offset + limit]

        return JSONResponse(content={
            "papers": papers_page,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list papers: {str(e)}")


@app.get("/papers/{paper_id}", tags=["Papers"])
async def get_paper(paper_id: str):
    """Get detailed information about a specific paper"""
    try:
        papers_file = Path("data/papers_info.json")
        if not papers_file.exists():
            raise HTTPException(status_code=404, detail="Papers database not found")

        with open(papers_file, 'r', encoding='utf-8') as f:
            papers_data = json.load(f)

        # Find paper
        paper = None
        if isinstance(papers_data, dict):
            paper = papers_data.get(paper_id)
        else:
            paper = next((p for p in papers_data if p.get('paper_id') == paper_id), None)

        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper {paper_id} not found")

        return JSONResponse(content=paper)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get paper: {str(e)}")


@app.get("/papers/suggest/questions", tags=["Papers"])
async def suggest_questions(limit: int = Query(5, ge=1, le=20)):
    """
    Suggest sample questions based on indexed papers

    Returns common question templates populated with paper topics
    """
    try:
        # Load papers to extract topics
        papers_file = Path("data/papers_info.json")
        if not papers_file.exists():
            # Return default suggestions
            return JSONResponse(content={
                "suggestions": [
                    "What is the transformer architecture?",
                    "How does BERT differ from GPT?",
                    "What are the key innovations in ResNet?",
                    "Explain the attention mechanism",
                    "What is transfer learning in NLP?"
                ][:limit]
            })

        with open(papers_file, 'r', encoding='utf-8') as f:
            papers_data = json.load(f)

        # Extract papers list
        if isinstance(papers_data, dict):
            papers_list = list(papers_data.values())
        else:
            papers_list = papers_data

        # Generate suggestions based on popular papers
        suggestions = []

        # Template questions
        templates = [
            "What is {}?",
            "How does {} work?",
            "What are the key innovations in {}?",
            "Explain the main contribution of {}",
            "What is the difference between {} and {}?"
        ]

        # Extract key concepts from titles
        concepts = []
        for paper in papers_list[:10]:  # Top 10 papers
            title = paper.get('title', '')
            # Extract main concept (simple heuristic)
            if ':' in title:
                concept = title.split(':')[0].strip()
            else:
                # Take first few meaningful words
                words = title.split()[:4]
                concept = ' '.join(words)
            concepts.append(concept)

        # Generate questions
        for i, template in enumerate(templates[:limit]):
            if '{}' in template:
                count = template.count('{}')
                if count == 1 and concepts:
                    suggestions.append(template.format(concepts[i % len(concepts)]))
                elif count == 2 and len(concepts) >= 2:
                    suggestions.append(template.format(concepts[i % len(concepts)], concepts[(i+1) % len(concepts)]))

        return JSONResponse(content={"suggestions": suggestions[:limit]})

    except Exception as e:
        # Return safe defaults on error
        return JSONResponse(content={
            "suggestions": [
                "What is the transformer architecture?",
                "How does attention mechanism work?",
                "What are the main differences between BERT and GPT?",
                "Explain residual connections in neural networks",
                "What is self-attention?"
            ][:limit]
        })


# ============================================================================
# Configuration Management Endpoints
# ============================================================================

@app.get("/config", tags=["Configuration"])
async def get_config():
    """Get current RAG configuration"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        status = rag_pipeline.get_status()
        config_dict = status.get('config', {})

        return JSONResponse(content={
            "config": config_dict,
            "editable_params": [
                "top_k",
                "vector_weight",
                "bm25_weight",
                "enable_reranking",
                "enable_diversification",
                "llm_temperature",
                "max_tokens"
            ]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


@app.put("/config", tags=["Configuration"])
async def update_config(request: ConfigUpdateRequest):
    """
    Update RAG configuration (hot reload without restart)

    Only specified parameters will be updated, others remain unchanged
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Update only provided fields
        config_updates = request.dict(exclude_none=True)

        if not config_updates:
            raise HTTPException(status_code=400, detail="No configuration updates provided")

        # Update RAG pipeline config
        for key, value in config_updates.items():
            if hasattr(rag_pipeline.config, key):
                setattr(rag_pipeline.config, key, value)

        # Return updated config
        updated_config = rag_pipeline.config.__dict__

        return JSONResponse(content={
            "message": "Configuration updated successfully",
            "updated_fields": list(config_updates.keys()),
            "current_config": updated_config
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@app.post("/config/reset", tags=["Configuration"])
async def reset_config():
    """Reset configuration to default values"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Reset to default config
        default_config = RAGConfig.from_global_config() if USE_CONFIG else RAGConfig()
        rag_pipeline.config = default_config

        return JSONResponse(content={
            "message": "Configuration reset to defaults",
            "config": default_config.__dict__
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset config: {str(e)}")


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "type": type(exc).__name__
        }
    )


# ============================================================================
# Run (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = 8000 if not USE_CONFIG else config.api.port
    host = "0.0.0.0" if not USE_CONFIG else config.api.host

    print(f"🚀 Starting server on {host}:{port}")
    print(f"📖 API docs: http://{host}:{port}/docs")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )

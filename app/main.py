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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time
import sys
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

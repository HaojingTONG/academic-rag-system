# 🎓 Academic RAG System

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-orange.svg)](https://github.com/HaojingTong/academic-rag-system)

An advanced Retrieval-Augmented Generation (RAG) system designed for academic paper question answering. Built with FastAPI backend, React frontend, and powered by state-of-the-art embedding models and LLMs.

> **✨ Perfect for**: AI researchers, students, and anyone who needs to quickly understand and query academic papers.

## 📑 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Development](#-development)
- [Configuration](#-configuration)
- [Advanced Features](#-advanced-features)
- [Documentation](#-documentation)

## 🌟 Features

### Core Capabilities

- **🔍 Intelligent Retrieval**: Hybrid search combining semantic vector search (BAAI/bge-m3) with BM25 keyword matching
- **🧠 Flexible LLM Integration**: Support for OpenAI GPT-4o/GPT-4o-mini and local Ollama models (Llama 3.1)
- **📊 Advanced Reranking**: Multi-stage reranking with BAAI/bge-reranker-large for optimal result quality
- **🎯 Citation Generation**: Automatic source attribution with confidence scores
- **📄 Comprehensive PDF Processing**: Full-text extraction with 15+ section types recognition
- **⚡ High Performance**: Optimized for Apple Silicon (M1/M2/M3) with MPS acceleration support

### Web Dashboard

- **💬 Ask Panel**: Interactive question-answering interface with real-time processing visualization
- **📚 Ingest Panel**: Easy paper upload and indexing management
- **🔧 Debug Panel**: System diagnostics and performance monitoring
- **📈 Processing Steps**: Real-time pipeline progress tracking

### Production Ready

- **🐳 Docker Support**: One-command deployment with docker-compose
- **🔐 Secure Configuration**: Environment-based configuration management
- **📊 Health Monitoring**: Built-in health checks and system statistics
- **🧪 Comprehensive Testing**: Unit, integration, and regression test suites
- **🚀 CI/CD Ready**: GitHub Actions workflow included

## 🚀 Quick Start

### Option 1: Docker (Recommended)

**Fastest way to get started - runs in 2 minutes!**

```bash
# 1. Clone the repository
git clone https://github.com/HaojingTong/academic-rag-system.git
cd academic-rag-system

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-key-here

# 3. Start the system
docker-compose up -d

# 4. Open your browser
# Backend API: http://localhost:8000
# Frontend Dashboard: http://localhost:3000
# API Documentation: http://localhost:8000/docs
```

### Option 2: Local Development

**For development and customization:**

```bash
# 1. Clone and setup
git clone https://github.com/HaojingTong/academic-rag-system.git
cd academic-rag-system

# 2. Create virtual environment
python3 -m venv venv_m3max
source venv_m3max/bin/activate  # On Windows: venv_m3max\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys and preferences

# 5. Index your papers (if you have PDFs in data/raw_papers/)
make index

# 6. Start backend
python app/main.py
# Backend runs on http://localhost:8000

# 7. Start frontend (in a new terminal)
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:3000
```

### Prerequisites

- **Python**: 3.9 or higher
- **Node.js**: 16+ (for frontend)
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 5GB+ free space
- **OpenAI API Key**: Get one at [platform.openai.com](https://platform.openai.com/api-keys)

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│  React + TypeScript + Tailwind (Port 3000)                  │
│  ┌────────────┬──────────────┬──────────────┐               │
│  │ Ask Panel  │ Ingest Panel │ Debug Panel  │               │
│  └────────────┴──────────────┴──────────────┘               │
└─────────────────────────────────────────────────────────────┘
                            ↓ HTTP/REST
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (Port 8000)               │
│  Endpoints: /query, /health, /stats, /docs                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Query Processing → Query Optimization              │   │
│  │ 2. Retrieval → Hybrid (Vector + BM25)                 │   │
│  │ 3. Reranking → BAAI/bge-reranker-large               │   │
│  │ 4. Composition → Context Assembly                      │   │
│  │ 5. Generation → OpenAI GPT-4o-mini / Ollama          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                                │
│  • Vector DB: ChromaDB (BAAI/bge-m3, 1024 dims)            │
│  • BM25 Index: Keyword search index                         │
│  • PDF Storage: Raw papers + parsed content                │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Backend**
- **API Framework**: FastAPI + Uvicorn
- **RAG Engine**: Custom pipeline with modular components
- **Embeddings**: BAAI/bge-m3 (1024-dim, multilingual)
- **Vector DB**: ChromaDB with persistent storage
- **LLM**: OpenAI GPT-4o-mini (primary), Ollama (optional)
- **Reranker**: BAAI/bge-reranker-large

**Frontend**
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite 5
- **Styling**: Tailwind CSS 3
- **HTTP Client**: Fetch API with timeout handling

**Infrastructure**
- **Containerization**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Testing**: pytest + pytest-cov
- **Code Quality**: black, ruff, mypy

### Project Structure

```
academic-rag-system/
├── app/                    # FastAPI web application
│   └── main.py            # API endpoints and server
├── rag/                    # Core RAG components
│   ├── pipeline.py        # Main RAG orchestration
│   ├── retriever.py       # Hybrid retrieval logic
│   ├── ranker.py          # Reranking and scoring
│   ├── composer.py        # Context composition
│   └── generator.py       # LLM integration
├── frontend/               # React web dashboard
│   ├── src/
│   │   ├── components/    # UI components
│   │   ├── hooks/         # Custom React hooks
│   │   └── utils/         # Utilities
│   └── package.json
├── configs/                # Configuration files
│   ├── config.yaml        # Main configuration
│   ├── development.yaml   # Dev environment
│   └── production.yaml    # Production settings
├── scripts/                # Utility scripts
│   ├── build_bm25_index.py
│   ├── evaluate_rag.py
│   └── diagnose_system.py
├── tests/                  # Test suites
│   ├── unit/
│   ├── integration/
│   └── regression/
├── docs/                   # Documentation
│   ├── setup/             # Setup guides
│   ├── summaries/         # Feature summaries
│   └── guides/            # User guides
├── data/                   # Data storage
│   ├── raw_papers/        # PDF files
│   └── bm25_index.pkl     # BM25 search index
├── vector_db/              # ChromaDB storage
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Backend container
├── Makefile               # Development commands
└── pyproject.toml         # Python project config
```

## 📖 Usage

### Web Dashboard

1. **Start the system** (see Quick Start)
2. **Open browser** to `http://localhost:3000`
3. **Ask questions** in the Ask Panel
4. **Upload papers** via the Ingest Panel
5. **Monitor system** in the Debug Panel

### Using the Makefile

The Makefile provides convenient commands for common tasks:

```bash
# Development
make bootstrap          # Full development setup
make install           # Install dependencies
make test              # Run all tests
make lint              # Check code quality
make format            # Format code

# Data Management
make ingest FILE=/path/to/paper.pdf  # Add a paper
make index             # Index new/modified papers
make reindex           # Full reindex (all papers)
make stats             # Show index statistics

# Running
make serve             # Start backend API
make smoke             # Run smoke tests
make watch             # Watch for changes and auto-index

# Cleaning
make clean             # Clean generated files
make clean-data        # Clean cached data
make clean-all         # Clean everything
```

### API Usage

**Query the RAG system:**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the transformer architecture?",
    "top_k": 5,
    "enable_reranking": true
  }'
```

**Check system health:**

```bash
curl http://localhost:8000/health
# Returns: {"status":"healthy","version":"2.0.0","rag_available":true}
```

**Get system statistics:**

```bash
curl http://localhost:8000/stats
```

### Python API

```python
from rag import RAGPipeline, RAGConfig

# Initialize the pipeline
config = RAGConfig()
pipeline = RAGPipeline(config)
pipeline.initialize()

# Query the system
result = pipeline.query(
    query="What are attention mechanisms?",
    top_k=5,
    enable_reranking=True
)

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} papers")
for i, source in enumerate(result['sources'], 1):
    print(f"{i}. {source['title']} (score: {source['score']:.2f})")
```

## 🔌 API Reference

### Endpoints

#### `POST /query`
Query the RAG system with a question.

**Request Body:**
```json
{
  "question": "string",
  "top_k": 5,
  "enable_reranking": true,
  "enable_diversification": true,
  "include_metadata": true
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [
    {
      "index": 1,
      "content": "string",
      "title": "string",
      "score": 0.95,
      "metadata": {...}
    }
  ],
  "num_sources": 5,
  "success": true,
  "metadata": {
    "retrieval_time": 0.15,
    "generation_time": 2.3,
    "total_time": 2.5
  }
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "rag_available": true
}
```

#### `GET /stats`
System statistics.

**Response:**
```json
{
  "num_documents": 68000,
  "num_papers": 59,
  "embedding_model": "BAAI/bge-m3",
  "llm_backend": "openai",
  "llm_model": "gpt-4o-mini"
}
```

#### `GET /docs`
Interactive API documentation (Swagger UI).

## 🛠️ Development

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/HaojingTong/academic-rag-system.git
cd academic-rag-system

# Run bootstrap script (installs everything)
make bootstrap

# Or manual setup
make venv              # Create virtual environment
source venv_m3max/bin/activate
make dev               # Install dev dependencies
```

### Running Tests

```bash
# Run all tests with coverage
make test

# Run specific test types
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-fast         # Skip slow tests

# Run smoke tests
make smoke
```

### Code Quality

```bash
# Format code
make format            # Auto-format with black & ruff

# Check formatting
make format-check      # Check without changes

# Lint code
make lint              # Run ruff and mypy

# Fix linting issues
make lint-fix          # Auto-fix issues
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start dev server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Type check
npm run type-check

# Lint
npm run lint
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

**Key settings:**

```bash
# LLM Configuration
LLM_BACKEND=openai              # or "ollama"
LLM_MODEL=gpt-4o-mini          # or "llama3.1:8b" for Ollama
OPENAI_API_KEY=sk-your-key     # Required for OpenAI

# Embedding Model
EMBEDDING_MODEL=BAAI/bge-m3    # High-quality multilingual
EMBEDDING_DEVICE=auto          # auto, cuda, mps, cpu

# Retrieval Settings
RETRIEVAL_TOP_K=5
RETRIEVAL_RERANK=true
RETRIEVAL_MMR_DIVERSITY=0.3

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Feature Flags
ENABLE_CITATIONS=true
ENABLE_FACT_CHECK=true
ENABLE_QUERY_EXPANSION=true
```

### Configuration Files

**Main config:** `configs/config.yaml`
```yaml
embedding:
  model: "BAAI/bge-m3"
  dimension: 1024

vector_store:
  collection_name: "papers_bge-m3_1024_v1"
  persist_directory: "./vector_db"

retrieval:
  top_k: 5
  enable_reranking: true

generation:
  temperature: 0.1
  max_tokens: 2000
```

**Environment-specific configs:**
- `configs/development.yaml` - Development settings (debug logging, hot reload)
- `configs/production.yaml` - Production settings (optimized performance, security)

### Using Ollama (Local LLM)

If you prefer running a local LLM instead of using OpenAI:

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Download a model
ollama pull llama3.1:8b

# 3. Update .env
LLM_BACKEND=ollama
LLM_MODEL=llama3.1:8b
OLLAMA_HOST=http://localhost:11434

# 4. Start Ollama service
ollama serve
```

## 🚀 Advanced Features

### Hybrid Retrieval

The system uses a two-stage retrieval approach:

1. **Stage 1: Parallel Search**
   - **Vector Search**: Semantic similarity using BAAI/bge-m3 embeddings
   - **BM25 Search**: Keyword-based retrieval for exact term matching

2. **Stage 2: Fusion & Reranking**
   - Results are fused using Reciprocal Rank Fusion (RRF)
   - Reranked using BAAI/bge-reranker-large for optimal relevance

```python
# Example: Custom retrieval configuration
result = pipeline.query(
    query="attention mechanisms in transformers",
    top_k=10,
    retrieval_strategy="hybrid",  # or "vector_only", "bm25_only"
    enable_reranking=True,
    diversity_factor=0.3  # MMR diversity (0=similarity, 1=diversity)
)
```

### Citation & Source Attribution

Every answer includes:
- **Source documents** with relevance scores
- **Direct quotes** from papers
- **Paper metadata** (title, authors, publication)
- **Confidence scores** for each source

### Advanced PDF Processing

The system intelligently processes academic PDFs with:

- **15+ Section Types**: Abstract, Introduction, Methods, Results, Discussion, etc.
- **Bilingual Support**: English and Chinese academic papers
- **Layout Parsing**: Handles single and double-column layouts (CVPR, ICLR, etc.)
- **Structure Preservation**: Maintains formatting, lists, formulas, and code blocks
- **Short Section Protection**: Preserves important short sections (conclusions, acknowledgments)
- **Feature Detection**: Identifies formulas, code, citations, and figures

### Query Optimization

Automatic query enhancement includes:
- **Query expansion** with synonyms and related terms
- **Multi-query generation** for complex questions
- **Context-aware reformulation** based on query type
- **Domain-specific optimization** for academic terminology

### Performance Monitoring

Access real-time metrics via:
- **Web Dashboard**: Debug Panel shows system status
- **API Endpoint**: `GET /stats` for programmatic access
- **Health Checks**: `GET /health` for monitoring

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

### Quick Start Guides
- **[Quick Start Guide](docs/setup/QUICK_START.md)** - Get up and running in 5 minutes
- **[OpenAI Setup](docs/setup/OPENAI_SETUP.md)** - Configure OpenAI API
- **[Docker Deployment](docs/setup/)** - Production deployment with Docker

### Feature Documentation
- **[RAG Pipeline](docs/summaries/RAG_OPTIMIZATION_SUMMARY.md)** - How the RAG system works
- **[Generation Quality](docs/summaries/GENERATION_OPTIMIZATION_SUMMARY.md)** - Answer generation details
- **[Hybrid Retrieval](docs/guides/)** - Advanced retrieval strategies

### Architecture & Design
- **[ADR: RAG Optimization](docs/ADR_RAG_OPTIMIZATION.md)** - Architecture decision record
- **[Project Structure](docs/refactoring/)** - Codebase organization

## 🔍 Troubleshooting

### Common Issues

**Backend won't start:**
```bash
# Check if port 8000 is in use
lsof -ti:8000 | xargs kill -9

# Verify configuration
python -c "from rag import RAGPipeline; print('✅ RAG available')"

# Check logs
tail -f logs/rag.log
```

**Frontend can't connect:**
```bash
# Verify backend is running
curl http://localhost:8000/health

# Check frontend environment
cat frontend/.env.local
# Should have: VITE_RAG_BASE_URL=http://localhost:8000
```

**OpenAI API errors:**
```bash
# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Check .env configuration
grep OPENAI_API_KEY .env
```

**Docker issues:**
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs backend
docker-compose logs frontend

# Restart services
docker-compose restart
```

### Getting Help

- **📝 Issues**: [GitHub Issues](https://github.com/HaojingTong/academic-rag-system/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/HaojingTong/academic-rag-system/discussions)
- **📧 Email**: haojing.tong@outlook.com

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Test** thoroughly (`make test`)
5. **Format** code (`make format`)
6. **Commit** with clear messages
7. **Push** to your fork
8. **Create** a Pull Request

### Development Standards

- Follow PEP 8 style guide
- Add type hints to all functions
- Write docstrings for public APIs
- Include unit tests for new features
- Update documentation as needed
- Maintain test coverage >80%

## 📈 Performance

### Benchmarks (Apple M3 Max)

| Metric | Performance |
|--------|-------------|
| 🔍 Retrieval latency | <200ms |
| 🧠 End-to-end query | 2-5s |
| 💾 Memory usage | 3-6GB |
| 📚 Indexed papers | 59+ |
| 📄 Document chunks | 68,000+ |

### RAG Quality Metrics

| Dimension | Score | Grade |
|-----------|-------|-------|
| 🎯 Overall | 0.72-0.85 | Good-Excellent |
| 🔍 Context Relevance | 0.68-0.82 | High |
| 🤝 Answer Faithfulness | 0.75-0.88 | Very High |
| 💡 Answer Relevance | 0.70-0.85 | High |
| 📈 Context Precision | 0.65-0.80 | Good |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds on excellent open-source work:

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://react.dev/) - UI library
- [sentence-transformers](https://www.sbert.net/) - Embedding models
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [OpenAI](https://openai.com/) - LLM API
- [Ollama](https://ollama.com/) - Local LLM runtime
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing

## 📞 Contact

- **Author**: Haojing Tong
- **Email**: haojing.tong@outlook.com
- **GitHub**: [@HaojingTong](https://github.com/HaojingTong)

---

<div align="center">

⭐ **If this project helps your research, please give it a star!** ⭐

[![GitHub stars](https://img.shields.io/github/stars/HaojingTong/academic-rag-system.svg?style=social&label=Star)](https://github.com/HaojingTong/academic-rag-system)
[![GitHub forks](https://img.shields.io/github/forks/HaojingTong/academic-rag-system.svg?style=social&label=Fork)](https://github.com/HaojingTong/academic-rag-system/fork)

**💡 Making AI research more efficient, knowledge access more intelligent!**

</div>

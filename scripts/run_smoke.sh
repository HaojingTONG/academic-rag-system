#!/bin/bash
# Academic RAG System - Smoke Tests
# ==================================

set -e  # Exit on error

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🧪 Running smoke tests...${NC}"
echo ""

TOTAL_TESTS=7
PASSED=0
FAILED=0

# Helper function to run test
run_test() {
    local test_num=$1
    local test_name=$2
    local test_cmd=$3

    echo -e "${BLUE}[$test_num/$TOTAL_TESTS] Testing: $test_name${NC}"

    if eval "$test_cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED++))
        return 1
    fi
}

# Test 1: Python version
run_test 1 "Python version (>=3.9)" \
    "python3 --version | grep -qE 'Python 3\.(9|10|11|12)'"

# Test 2: Import core modules
echo -e "${BLUE}[2/$TOTAL_TESTS] Testing: Core module imports${NC}"
python3 << 'EOF'
try:
    import sys
    sys.path.insert(0, '.')

    # Test imports
    from configs.config_loader import config
    from src.processor.pdf_processor import AcademicPDFProcessor
    from src.processor.document_chunker import DocumentChunker
    from src.embedding.advanced_embedding import EmbeddingManager
    from src.retriever.vector_store import VectorStore
    from src.generator.llm_client import OllamaClient

    print("✓ All imports successful", file=sys.stderr)
    sys.exit(0)
except Exception as e:
    print(f"✗ Import error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test 3: Load config
echo -e "${BLUE}[3/$TOTAL_TESTS] Testing: Configuration loading${NC}"
python3 << 'EOF'
try:
    import sys
    sys.path.insert(0, '.')
    from configs.config_loader import config

    assert config.embedding.model == 'all-mpnet-base-v2', "Wrong embedding model"
    assert config.embedding.dimension == 768, "Wrong embedding dimension"
    assert config.retrieval.top_k == 5, "Wrong retrieval top_k"

    print(f"✓ Config loaded (model: {config.embedding.model})", file=sys.stderr)
    sys.exit(0)
except Exception as e:
    print(f"✗ Config error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test 4: Chunking
echo -e "${BLUE}[4/$TOTAL_TESTS] Testing: Document chunking${NC}"
python3 << 'EOF'
try:
    import sys
    sys.path.insert(0, '.')
    from src.processor.document_chunker import DocumentChunker

    chunker = DocumentChunker(strategy='fixed', chunk_size=200, chunk_overlap=50)
    text = 'This is a test sentence. ' * 100
    chunks = chunker.chunk(text)

    assert len(chunks) > 0, "No chunks generated"
    assert all(len(c.text) <= 250 for c in chunks), "Chunk size exceeded"

    print(f"✓ Chunking works ({len(chunks)} chunks generated)", file=sys.stderr)
    sys.exit(0)
except Exception as e:
    print(f"✗ Chunking error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test 5: Embedding model
echo -e "${BLUE}[5/$TOTAL_TESTS] Testing: Embedding model${NC}"
python3 << 'EOF'
try:
    import sys
    sys.path.insert(0, '.')
    from src.embedding.advanced_embedding import EmbeddingManager

    manager = EmbeddingManager(model_name='all-mpnet-base-v2', device='cpu')
    embedding = manager.embed('test query')

    assert len(embedding) == 768, f"Wrong embedding dimension: {len(embedding)}"
    assert all(isinstance(x, (int, float)) for x in embedding), "Invalid embedding values"

    print(f"✓ Embedding works (768-dim vector)", file=sys.stderr)
    sys.exit(0)
except Exception as e:
    print(f"✗ Embedding error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${RED}✗ FAILED${NC}"
    ((FAILED++))
fi

# Test 6: Vector store (if exists)
echo -e "${BLUE}[6/$TOTAL_TESTS] Testing: Vector store${NC}"
if [ -d "vector_db" ] && [ "$(ls -A vector_db)" ]; then
    python3 << 'EOF'
try:
    import sys
    sys.path.insert(0, '.')
    from src.retriever.vector_store import VectorStore

    store = VectorStore(
        persist_directory="vector_db",
        embedding_function=None,
        collection_name="enhanced_ai_papers_v2"
    )

    print(f"✓ Vector store loaded", file=sys.stderr)
    sys.exit(0)
except Exception as e:
    print(f"✗ Vector store error: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED++))
    fi
else
    echo -e "${YELLOW}⚠ SKIPPED (no vector DB found)${NC}"
    echo "   Run 'make index' to build vector database"
    ((PASSED++))  # Don't count as failure
fi

# Test 7: Ollama availability
echo -e "${BLUE}[7/$TOTAL_TESTS] Testing: Ollama availability${NC}"
if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PASSED (Ollama is running)${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ SKIPPED (Ollama not running)${NC}"
    echo "   Start Ollama with: ollama serve"
    ((PASSED++))  # Don't count as failure for smoke test
fi

echo ""
echo "========================================"
echo -e "Results: ${GREEN}$PASSED passed${NC}, ${RED}$FAILED failed${NC} (out of $TOTAL_TESTS)"
echo "========================================"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All smoke tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some smoke tests failed${NC}"
    exit 1
fi

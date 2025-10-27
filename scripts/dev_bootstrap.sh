#!/bin/bash
# Academic RAG System - Development Bootstrap Script
# ===================================================

set -e  # Exit on error

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Academic RAG System - Development Bootstrap${NC}"
echo "================================================"
echo ""

# Check Python version
echo -e "${BLUE}📌 Checking Python version...${NC}"
python3 --version | grep -qE "Python 3\.(9|10|11|12)" || {
    echo -e "${RED}❌ Python 3.9+ required${NC}"
    echo "Current version: $(python3 --version)"
    exit 1
}
echo -e "${GREEN}✓ Python $(python3 --version | cut -d' ' -f2) detected${NC}"
echo ""

# Check if virtual environment exists
VENV="venv_m3max"
if [ ! -d "$VENV" ]; then
    echo -e "${BLUE}📦 Creating virtual environment...${NC}"
    python3 -m venv $VENV
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source $VENV/bin/activate

# Upgrade pip
echo -e "${BLUE}📥 Upgrading pip...${NC}"
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ Pip upgraded${NC}"
echo ""

# Install production dependencies
echo -e "${BLUE}📦 Installing production dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Production dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  requirements.txt not found${NC}"
fi
echo ""

# Install development dependencies
echo -e "${BLUE}📦 Installing development dependencies...${NC}"
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt --quiet
    echo -e "${GREEN}✓ Development dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️  requirements-dev.txt not found${NC}"
fi
echo ""

# Setup configuration
echo -e "${BLUE}⚙️  Setting up configuration...${NC}"
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env file from .env.example${NC}"
        echo -e "${YELLOW}  → Please customize .env for your environment${NC}"
    else
        echo -e "${YELLOW}⚠️  .env.example not found${NC}"
    fi
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi
echo ""

# Create necessary directories
echo -e "${BLUE}📁 Creating directories...${NC}"
mkdir -p logs
mkdir -p data/raw_papers
mkdir -p data/embedding_cache
mkdir -p data/evaluation
mkdir -p vector_db
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check Ollama
echo -e "${BLUE}🤖 Checking Ollama...${NC}"
if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    OLLAMA_VERSION=$(curl -s http://localhost:11434/api/version | python3 -c "import sys, json; print(json.load(sys.stdin)['version'])" 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓ Ollama is running (version: $OLLAMA_VERSION)${NC}"
else
    echo -e "${YELLOW}⚠️  Ollama not running${NC}"
    echo "   Start Ollama with: ollama serve"
    echo "   Or install from: https://ollama.ai"
fi
echo ""

# Download embedding model
echo -e "${BLUE}📊 Checking embedding model...${NC}"
python3 -c "
from sentence_transformers import SentenceTransformer
import sys
try:
    model = SentenceTransformer('all-mpnet-base-v2')
    print('${GREEN}✓ Embedding model available${NC}', file=sys.stderr)
except Exception as e:
    print(f'${RED}✗ Error loading embedding model: {e}${NC}', file=sys.stderr)
    sys.exit(1)
" 2>&1
echo ""

# Run smoke tests
echo -e "${BLUE}🧪 Running smoke tests...${NC}"
if [ -f "scripts/run_smoke.sh" ]; then
    bash scripts/run_smoke.sh
else
    echo -e "${YELLOW}⚠️  Smoke test script not found${NC}"
fi
echo ""

# Summary
echo -e "${GREEN}✅ Bootstrap complete!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Activate environment: ${YELLOW}source $VENV/bin/activate${NC}"
echo "  2. Customize config: ${YELLOW}vi .env${NC}"
echo "  3. Check Ollama models: ${YELLOW}ollama list${NC}"
echo "  4. Build index: ${YELLOW}make index${NC}"
echo "  5. Run CLI: ${YELLOW}make run${NC}"
echo "  6. Run tests: ${YELLOW}make test${NC}"
echo "  7. Start API server: ${YELLOW}make serve${NC}"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo "  • ${YELLOW}make help${NC} - Show all available commands"
echo "  • ${YELLOW}make status${NC} - Show system status"
echo "  • ${YELLOW}make smoke${NC} - Run quick smoke tests"
echo ""
echo -e "${GREEN}Happy coding! 🎉${NC}"

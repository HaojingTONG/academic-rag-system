#!/bin/bash
# Incremental Index Refresh Script
# =================================
#
# Detect and process new/modified/deleted papers
# Usage: ./scripts/refresh_index.sh [--force]

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}📦 Incremental Index Refresh${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check if force flag provided
FORCE_FLAG=""
if [ "$1" = "--force" ]; then
    FORCE_FLAG="--force"
    echo -e "${YELLOW}⚠️  Force mode enabled (will reprocess all documents)${NC}"
    echo ""
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Activating virtual environment..."
    source venv/bin/activate
fi

# Run incremental indexer
echo -e "${GREEN}✓${NC} Running indexer..."
echo ""

python -m indexer refresh $FORCE_FLAG

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Index refresh complete!${NC}"
else
    echo ""
    echo -e "${RED}❌ Index refresh failed!${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Next steps:${NC}"
echo -e "  ${GREEN}•${NC} Run smoke test: make smoke"
echo -e "  ${GREEN}•${NC} Query the system: python app/main.py"
echo -e "${BLUE}======================================${NC}"

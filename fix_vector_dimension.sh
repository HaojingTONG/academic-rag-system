#!/bin/bash
# Fix Vector Dimension Mismatch Issue
# =====================================
# Problem: Vector DB has 768-dim embeddings, but model generates 1024-dim
# Solution: Backup old DB and reindex with current model (BAAI/bge-m3, 1024-dim)

set -e  # Exit on error

echo "🔧 Fixing Vector Dimension Mismatch"
echo "===================================="
echo ""

# Check if we're in the right directory
if [ ! -d "vector_db" ]; then
    echo "❌ Error: vector_db directory not found"
    echo "   Please run this script from the project root"
    exit 1
fi

# Step 1: Backup old vector database
echo "📦 Step 1: Backing up old vector database..."
if [ -d "vector_db_backup_768dim" ]; then
    echo "   ⚠️  Backup already exists, skipping..."
else
    mv vector_db vector_db_backup_768dim
    echo "   ✓ Backed up to vector_db_backup_768dim"
fi

# Step 2: Create new vector database directory
echo ""
echo "📁 Step 2: Creating new vector database directory..."
mkdir -p vector_db
echo "   ✓ Created vector_db/"

# Step 3: Count papers
PAPER_COUNT=$(ls data/raw_papers/*.pdf 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "📄 Found $PAPER_COUNT PDF papers to index"

# Step 4: Reindex papers
echo ""
echo "🔄 Step 3: Reindexing papers with BAAI/bge-m3 (1024-dim)..."
echo "   This will take approximately $((PAPER_COUNT / 10)) minutes..."
echo ""

# Run indexing with --force flag to rebuild everything
if [ -f "scripts/refresh_index.sh" ]; then
    ./scripts/refresh_index.sh --force
elif [ -f "Makefile" ]; then
    make index
else
    echo "❌ Error: Cannot find indexing script"
    echo "   Please run manually:"
    echo "   python -m indexer refresh --force"
    exit 1
fi

echo ""
echo "===================================="
echo "✅ Fix Complete!"
echo "===================================="
echo ""
echo "What was done:"
echo "  1. ✓ Backed up old 768-dim vector database"
echo "  2. ✓ Created new vector database directory"
echo "  3. ✓ Reindexed $PAPER_COUNT papers with 1024-dim embeddings"
echo ""
echo "Next steps:"
echo "  1. Restart backend: python app/main.py"
echo "  2. Test query: curl -X POST http://localhost:8000/query \\"
echo "                   -H 'Content-Type: application/json' \\"
echo "                   -d '{\"question\": \"什么是AI?\"}'"
echo ""
echo "If you want to restore the old database:"
echo "  rm -rf vector_db"
echo "  mv vector_db_backup_768dim vector_db"
echo ""

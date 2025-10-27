# Migration Guide: RAG System v1.0 → v2.0

> **Status**: DRAFT - Ready for review
> **Version**: 2.0.0
> **Date**: 2024-01-15

---

## Table of Contents

1. [Overview](#overview)
2. [What's Changing](#whats-changing)
3. [Breaking Changes](#breaking-changes)
4. [Step-by-Step Migration](#step-by-step-migration)
5. [Import Path Updates](#import-path-updates)
6. [Configuration Migration](#configuration-migration)
7. [Testing Your Migration](#testing-your-migration)
8. [Rollback Instructions](#rollback-instructions)
9. [FAQ](#faq)

---

## Overview

This guide helps you migrate from RAG System v1.0 to the streamlined v2.0 architecture.

**Key Improvements**:
- ✅ Unified configuration (config.yaml + .env)
- ✅ Cleaner module organization
- ✅ Consolidated retrieval modules
- ✅ FastAPI web API (optional)
- ✅ Formal pytest test framework
- ✅ Makefile for common tasks

**Migration Time**: ~15 minutes (for most users)

---

## What's Changing

### Directory Structure

```
OLD STRUCTURE                    NEW STRUCTURE
─────────────────────────────────────────────────────
scripts/main_rag_system.py   →   app/cli.py (+ app/main.py for API)

src/processor/               →   indexer/
  ├── pdf_processor.py       →     ├── ingest.py
  ├── document_chunker.py    →     └── chunking.py

src/retriever/               →   rag/
  ├── vector_store.py        →     ├── retriever.py (merged)
  ├── enhanced_*.py          →     └── ranker.py
  └── advanced_retrieval.py

src/generator/               →   rag/
  ├── llm_client.py          →     ├── generator.py
  ├── prompt_engineering.py  →     ├── prompt.py
  └── quality_enhancement.py →     └── evaluator.py

src/embedding/               →   models/
  └── advanced_embedding.py  →     └── embed_client.py

src/evaluation/              →   rag/evaluator.py (merged)

[no config files]            →   configs/
                                    ├── config.yaml (NEW!)
                                    └── logging.yaml

[scattered tests]            →   tests/
                                    ├── unit/
                                    ├── integration/
                                    └── conftest.py
```

### Deleted Files

The following duplicate files have been **removed**:
- ❌ `src/processor/pdf_processor 2.py`
- ❌ `src/processor/__init__ 2.py`

These were old duplicates and are safely backed up in `.backup/pre-refactor/`.

---

## Breaking Changes

### 1. Import Paths

**Old imports will still work** (with deprecation warnings) but should be updated:

```python
# ❌ OLD (deprecated, but still works)
from src.retriever.vector_store import VectorStore
from src.generator.llm_client import OllamaClient
from src.processor.document_chunker import DocumentChunker

# ✅ NEW (recommended)
from rag.retriever import VectorStore
from rag.generator import OllamaClient
from indexer.chunking import DocumentChunker
```

### 2. Configuration

**Old**: Hardcoded values in each module
```python
chunk_size = 600  # hardcoded
top_k = 5  # hardcoded
```

**New**: Centralized configuration
```python
from configs import config
chunk_size = config.chunking.chunk_size
top_k = config.retrieval.top_k
```

### 3. CLI Entry Point

**Old**:
```bash
python scripts/main_rag_system.py
```

**New**:
```bash
# Method 1: Using Makefile
make run

# Method 2: Direct CLI
python -m app.cli query --interactive

# Method 3: Old way (still works with warning)
python scripts/main_rag_system.py
```

---

## Step-by-Step Migration

### Step 1: Backup Your Current State

```bash
# Create a git tag for easy rollback
git tag pre-refactor-backup

# Or create a manual backup
cp -r . ../rag-system-backup
```

### Step 2: Pull/Checkout the New Code

```bash
# If using git
git checkout refactor/streamline-architecture

# Or download the refactored version
```

### Step 3: Update Configuration

**A. Copy the example config:**
```bash
cp .env.example .env
```

**B. Migrate your settings:**

If you had custom settings in code, move them to `.env`:

```bash
# .env
EMBEDDING_MODEL=all-mpnet-base-v2
OLLAMA_MODEL=llama3.1:8b
RETRIEVAL_TOP_K=5
GENERATION_TEMPERATURE=0.1
```

**C. Review config.yaml:**

Edit `configs/config.yaml` to customize:
- Model names
- Chunk sizes
- Retrieval parameters
- API settings (if using web API)

### Step 4: Run Bootstrap Script

```bash
# This will:
# - Check Python version
# - Install dependencies
# - Download embedding models
# - Run smoke tests
bash scripts/dev_bootstrap.sh
```

### Step 5: Verify Installation

```bash
# Run smoke tests
make smoke

# Check system status
make status

# Expected output:
# ✅ All smoke tests passed!
```

### Step 6: Rebuild Vector Index (if needed)

If vector DB format changed or if you want a clean start:

```bash
# Backup old vector DB
mv vector_db vector_db.old

# Rebuild index
make index
```

**Note**: This can take 10-30 minutes depending on the number of papers.

### Step 7: Update Your Code

**A. Update imports:**

Run this helper script to update imports automatically:

```bash
# Coming soon: automatic import updater
python scripts/update_imports.py
```

Or manually update using find/replace:

```bash
# Example: Update retriever imports
grep -rl "from src.retriever" . | xargs sed -i '' 's/from src.retriever/from rag.retriever/g'
```

**B. Update custom scripts:**

If you have custom scripts that import RAG system modules:

1. Update import paths (see [Import Path Updates](#import-path-updates))
2. Update configuration loading:

```python
# OLD
embedding_model = "all-mpnet-base-v2"
chunk_size = 600

# NEW
from configs import config
embedding_model = config.embedding.model
chunk_size = config.chunking.chunk_size
```

### Step 8: Test Everything

```bash
# Run all tests
make test

# Or run selectively
make test-unit          # Fast unit tests
make test-integration   # Integration tests
make test-regression    # Ensure no performance degradation
```

### Step 9: Update Your Workflow

**Old workflow:**
```bash
python scripts/process_pdf_fulltext.py
python scripts/main_rag_system.py
python scripts/evaluate_rag_system.py
```

**New workflow:**
```bash
make index      # Build/update index
make run        # Run CLI
make evaluate   # Run evaluation
```

See `make help` for all available commands.

---

## Import Path Updates

### Complete Import Mapping

| Old Import | New Import | Status |
|------------|------------|--------|
| `src.retriever.vector_store` | `rag.retriever` | ⚠️ Deprecated |
| `src.retriever.enhanced_vector_store` | `rag.retriever` | ⚠️ Deprecated |
| `src.retriever.advanced_retrieval` | `rag.retriever` | ⚠️ Deprecated |
| `src.generator.llm_client` | `rag.generator` | ⚠️ Deprecated |
| `src.generator.prompt_engineering` | `rag.prompt` | ⚠️ Deprecated |
| `src.generator.quality_enhancement` | `rag.evaluator` | ⚠️ Deprecated |
| `src.processor.pdf_processor` | `indexer.ingest` | ⚠️ Deprecated |
| `src.processor.document_chunker` | `indexer.chunking` | ⚠️ Deprecated |
| `src.embedding.advanced_embedding` | `models.embed_client` | ⚠️ Deprecated |
| `src.evaluation.rag_evaluator` | `rag.evaluator` | ⚠️ Deprecated |

### Compatibility Shims

**Good news**: Old imports still work for now!

The system includes compatibility shims that redirect old imports to new locations with deprecation warnings:

```python
# This still works but shows a warning
from src.retriever.vector_store import VectorStore
# DeprecationWarning: src.retriever is deprecated, use rag.retriever

# Update to:
from rag.retriever import VectorStore
```

**Shims will be removed in v2.1.0** (estimated 3 months), so please update your code!

---

## Configuration Migration

### Finding Hardcoded Values

Your old code might have hardcoded configuration values. Here's how to find and migrate them:

```bash
# Find hardcoded chunk sizes
grep -rn "chunk_size.*=" src/

# Find hardcoded model names
grep -rn "all-mpnet" src/

# Find hardcoded top_k values
grep -rn "top_k.*=" src/
```

### Migration Examples

#### Example 1: Chunking Configuration

**Before:**
```python
# src/processor/document_chunker.py
class DocumentChunker:
    def __init__(self):
        self.chunk_size = 600  # hardcoded
        self.chunk_overlap = 100  # hardcoded
```

**After:**
```python
# indexer/chunking.py
from configs import config

class DocumentChunker:
    def __init__(self):
        self.chunk_size = config.chunking.chunk_size
        self.chunk_overlap = config.chunking.chunk_overlap
```

#### Example 2: Retrieval Configuration

**Before:**
```python
# Custom script
retriever = HybridRetriever(
    top_k=5,
    bm25_weight=0.3,
    vector_weight=0.7
)
```

**After:**
```python
# Custom script
from configs import config

retriever = HybridRetriever(
    top_k=config.retrieval.top_k,
    bm25_weight=config.retrieval.bm25_weight,
    vector_weight=config.retrieval.vector_weight
)

# Or use defaults from config
retriever = HybridRetriever()  # Automatically uses config
```

#### Example 3: LLM Configuration

**Before:**
```python
llm_client = OllamaClient(
    model="llama3.1:8b",
    temperature=0.1,
    max_tokens=2000
)
```

**After:**
```python
from configs import config

llm_client = OllamaClient(
    model=config.generation.model,
    temperature=config.generation.temperature,
    max_tokens=config.generation.max_tokens
)

# Or use environment variables
# Set in .env: OLLAMA_MODEL=llama3.1:8b
llm_client = OllamaClient()  # Automatically uses config
```

---

## Testing Your Migration

### 1. Smoke Tests

```bash
make smoke
```

**Expected output:**
```
[1/7] Testing: Python version (>=3.9)            ✓ PASSED
[2/7] Testing: Core module imports               ✓ PASSED
[3/7] Testing: Configuration loading             ✓ PASSED
[4/7] Testing: Document chunking                 ✓ PASSED
[5/7] Testing: Embedding model                   ✓ PASSED
[6/7] Testing: Vector store                      ✓ PASSED
[7/7] Testing: Ollama availability               ✓ PASSED

✅ All smoke tests passed!
```

### 2. Unit Tests

```bash
make test-unit
```

Should pass all tests in `tests/unit/`.

### 3. Integration Tests

```bash
make test-integration
```

Tests the full RAG pipeline end-to-end.

### 4. Regression Tests

```bash
make test-regression
```

Ensures performance hasn't degraded:
- Retrieval precision ≥ baseline
- Generation quality ≥ baseline
- Latency ≤ baseline

### 5. Manual Testing

```bash
# Test CLI
make run

# Try a query
> query: What is the transformer architecture?

# Check output quality
```

---

## Rollback Instructions

If something goes wrong, here's how to rollback:

### Method 1: Git Tag Rollback

```bash
# Return to pre-refactor state
git checkout pre-refactor-backup

# Restore virtual environment
source venv_m3max/bin/activate

# Verify it works
python scripts/main_rag_system.py
```

### Method 2: Manual Rollback

```bash
# Restore from backup
rm -rf your-project-dir
cp -r ../rag-system-backup your-project-dir
cd your-project-dir

# Restore vector DB (if backed up)
rm -rf vector_db
mv vector_db.old vector_db
```

### Method 3: Revert Git Commits

```bash
# Find the commit hash before refactor
git log --oneline

# Revert to that commit
git reset --hard <commit-hash>

# Or merge from main branch
git checkout main
```

### After Rollback

```bash
# Verify old system works
python scripts/main_rag_system.py

# Run old tests (if any)
python test_*.py
```

---

## FAQ

### Q: Do I need to rebuild my vector database?

**A**: Not required, but recommended if:
- You want to use new chunking strategies
- You've updated papers in `data/raw_papers/`
- You're experiencing issues with retrieval

The vector DB format hasn't changed, so old indices should work.

### Q: Will my old scripts break?

**A**: No, old imports still work via compatibility shims. You'll see deprecation warnings, but functionality is preserved.

Update your imports before v2.1.0 (3 months from now).

### Q: How do I use the new FastAPI web API?

**A**: Enable in config:

```yaml
# configs/config.yaml
features:
  enable_web_api: true
```

Then run:
```bash
make serve
```

Access at http://localhost:8000/docs

### Q: Can I use both old and new import styles?

**A**: Yes, during the migration period. But pick one style per project to avoid confusion.

### Q: What if I have custom modifications to src/ modules?

**A**:
1. Copy your custom code to a safe location
2. Complete the migration
3. Re-apply your customizations to the new modules
4. Test thoroughly

### Q: How do I configure logging now?

**A**: Edit `configs/config.yaml`:

```yaml
logging:
  level: "INFO"  # or DEBUG, WARNING, ERROR
  handlers:
    console:
      enabled: true
      level: "INFO"
    file:
      enabled: true
      path: "logs/rag_system.log"
```

Or use environment variable:
```bash
export LOG_LEVEL=DEBUG
```

### Q: What happened to the evaluation scripts?

**A**: Now accessible via:

```bash
make evaluate
```

Or directly:
```bash
python -m rag.evaluator --dataset data/evaluation/test_dataset.json
```

### Q: Can I customize chunk size per query?

**A**: Yes, override config at runtime:

```python
from configs import config

# Temporarily override
original_size = config.chunking.chunk_size
config.chunking.chunk_size = 400

# Do your work
chunker = DocumentChunker()  # Uses 400

# Restore
config.chunking.chunk_size = original_size
```

### Q: How do I add a new configuration parameter?

**A**:

1. Add to `configs/config.yaml`:
```yaml
my_module:
  my_parameter: 123
```

2. Add dataclass in `configs/config_loader.py`:
```python
@dataclass
class MyModuleConfig:
    my_parameter: int = 123

@dataclass
class Config:
    # ... existing configs
    my_module: MyModuleConfig = field(default_factory=MyModuleConfig)
```

3. Use in code:
```python
from configs import config
value = config.my_module.my_parameter
```

### Q: Where are the logs now?

**A**: `logs/rag_system.log` (created automatically)

View with:
```bash
tail -f logs/rag_system.log
```

### Q: How do I run tests on a subset of code?

```bash
# Specific module
pytest tests/unit/test_retriever.py

# Specific test
pytest tests/unit/test_retriever.py::TestVectorStore::test_similarity_search

# By marker
pytest -m "not slow"
pytest -m integration
```

---

## Additional Resources

- **Refactoring Plan**: See `REFACTORING_PLAN.md` for technical details
- **API Documentation**: Run `make serve` and visit http://localhost:8000/docs
- **Makefile Commands**: Run `make help` to see all available commands
- **Configuration Reference**: See `configs/config.yaml` for all options

---

## Support

**Issues**: Please report migration problems at:
- GitHub Issues: https://github.com/your-org/academic-rag-system/issues
- Email: support@example.com

**Need Help?**
1. Check this guide's FAQ section
2. Run `make status` to diagnose issues
3. Review `logs/rag_system.log` for errors
4. Open an issue with:
   - Error message
   - Output of `make status`
   - Relevant logs

---

## Checklist

Use this checklist to track your migration:

- [ ] Backup current state (git tag or manual copy)
- [ ] Pull new code / checkout refactor branch
- [ ] Copy `.env.example` to `.env`
- [ ] Customize `.env` and `configs/config.yaml`
- [ ] Run `bash scripts/dev_bootstrap.sh`
- [ ] Run `make smoke` (all tests pass)
- [ ] Update import paths in custom scripts
- [ ] Rebuild vector index: `make index` (optional)
- [ ] Run `make test` (all tests pass)
- [ ] Test CLI: `make run`
- [ ] Test custom scripts still work
- [ ] Update documentation/README
- [ ] Commit changes
- [ ] Remove old backup (after confirming everything works)

---

**Document Version**: 1.0
**Last Updated**: 2024-01-15
**Status**: Ready for Use

---

*Happy Migrating! 🚀*

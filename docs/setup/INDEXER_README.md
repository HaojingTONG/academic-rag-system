# Incremental Indexing System

Automatic detection and processing of new/modified/deleted academic papers with vector database synchronization.

---

## 🚀 Quick Start

```bash
# 1. Add papers to data/raw_papers/
cp /path/to/paper.pdf data/raw_papers/

# 2. Index papers (incremental - only processes new/modified)
make index

# 3. Validate index quality
make smoke

# 4. Query the system
python app/main.py
```

---

## 📋 Features

### Incremental Processing
- **SHA256-based change detection**: Only reprocess when content changes
- **Idempotent upsert**: Safe to run multiple times
- **Soft delete support**: Deleted papers marked in manifest, removable from vector DB

### Document Processing Pipeline
```
PDF/TXT → Parse → Chunk → Embed → Upsert
         ↓        ↓       ↓       ↓
      Manifest  Manifest Manifest VectorDB
```

### Manifest Database
SQLite-based tracking with:
- Document ID, source path, SHA256
- Parse/chunk/embed/upsert timestamps
- Embedding model namespace versioning
- Soft/hard delete support

### Namespace Versioning
```
Format: {model}@{dimension}@{version}
Example: bge-m3@1024@v1

Benefits:
- Safe model migration
- A/B testing
- Rollback capability
```

---

## 🛠️ Usage

### Basic Operations

```bash
# Incremental index (only new/modified papers)
make index
# or
./scripts/refresh_index.sh

# Force full reindex
make reindex
# or
./scripts/refresh_index.sh --force

# View statistics
make stats

# Run smoke tests
make smoke
```

### Adding Papers

```bash
# Method 1: Copy to raw_papers/
cp paper.pdf data/raw_papers/
make index

# Method 2: Use make target
make ingest FILE=/path/to/paper.pdf
make index
```

### Deleting Papers

```bash
# Delete by document ID
make delete DOC=vaswani2017attention

# Or use Python API
python -c "
from indexer import IncrementalIndexer
indexer = IncrementalIndexer()
indexer.delete_document('doc_id', soft=True)
"
```

### Namespace Migration

```bash
# Migrate to new embedding model
make migrate NS=bge-m3@1024@v2

# This will:
# 1. Reprocess all documents
# 2. Generate new embeddings
# 3. Upsert to new collection
# 4. Keep old namespace for rollback
```

---

## 📁 Directory Structure

```
academic-rag-system/
├── indexer/
│   ├── __init__.py          # Main orchestrator
│   ├── metadata.py          # Manifest database
│   ├── parse.py             # PDF/TXT parser
│   ├── chunking.py          # Document chunking
│   ├── embed.py             # Embedding generation
│   └── upsert.py            # Vector DB operations
├── configs/
│   └── index.yaml           # Indexing configuration
├── scripts/
│   └── refresh_index.sh     # Index refresh script
├── data/
│   ├── raw_papers/          # Input: PDF/TXT files
│   ├── parsed/              # Parsed JSON documents
│   └── manifest.db          # SQLite manifest
├── vector_db/               # ChromaDB storage
└── eval/
    └── smoke_eval.py        # Index quality tests
```

---

## ⚙️ Configuration

### configs/index.yaml

```yaml
# Chunking
chunking:
  chunk_size: 700        # Target tokens per chunk
  overlap: 0.12          # 12% overlap
  add_context: true      # Add title/section prefix

# Embedding
embedding:
  model_namespace: "bge-m3@1024@v1"
  model_name: "BAAI/bge-m3"
  dimension: 1024
  batch_size: 32
  device: "auto"         # auto, cpu, cuda, mps

# Vector DB
vector_db:
  collection_template: "papers_{namespace}"
  distance: "cosine"
  index_type: "hnsw"

# Manifest
manifest:
  db_path: "data/manifest.db"
  soft_delete: true
```

---

## 🔍 Manifest Schema

```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,           -- Unique document ID
    source_path TEXT NOT NULL,         -- Path to source file
    sha256 TEXT NOT NULL,              -- Content hash
    title TEXT,                        -- Extracted title
    authors TEXT,                      -- Extracted authors
    year INTEGER,                      -- Publication year
    venue TEXT,                        -- Publication venue
    doi TEXT,                          -- DOI if available
    parsed_at TIMESTAMP,               -- Parse timestamp
    chunked_at TIMESTAMP,              -- Chunk timestamp
    embed_model_ns TEXT,               -- Embedding namespace
    upserted_at TIMESTAMP,             -- Upsert timestamp
    deleted_at TIMESTAMP,              -- Deletion timestamp
    is_active BOOLEAN DEFAULT 1,       -- Active flag
    created_at TIMESTAMP,              -- Creation time
    updated_at TIMESTAMP               -- Last update time
);
```

---

## 🧪 Testing

### Smoke Evaluation

Tests index quality with fixed queries:

```bash
make smoke
```

**Metrics:**
- `retrieved_n ≥ 3`: Minimum documents retrieved
- `kept_n ≥ 2`: Minimum after filtering
- Latency: p50, p95
- Citation presence

**Fails if:**
- Any query retrieves < 3 documents
- Any query keeps < 2 documents after filtering

### Manual Testing

```python
from indexer import IncrementalIndexer

# Initialize
indexer = IncrementalIndexer()

# Refresh index
indexer.refresh()

# Get statistics
indexer._print_stats()

# Delete document
indexer.delete_document('doc_id')
```

---

## 🔄 Workflow Examples

### Daily Incremental Update

```bash
# 1. Researchers drop papers in raw_papers/
cp new_paper.pdf data/raw_papers/

# 2. Run incremental index (cron or manual)
make index

# 3. Validate quality
make smoke

# 4. System automatically available for queries
```

### Model Migration

```bash
# Current: bge-m3@1024@v1
# Target:  bge-m3@1024@v2

# 1. Update config
vim configs/index.yaml  # Change model_namespace

# 2. Reindex with new namespace
make reindex

# 3. Validate new index
make smoke

# 4. If passed, update query path to use v2
# If failed, rollback to v1
```

### Cleanup Old Namespaces

```python
from indexer.upsert import VectorDBUpserter

upserter = VectorDBUpserter()

# List namespaces
stats = upserter.get_stats()
print(stats['namespaces'])

# Delete old namespace
upserter.delete_namespace('bge-m3@1024@v1')
```

---

## 📊 Monitoring

### Statistics

```bash
make stats
```

**Output:**
```
📊 Statistics:
────────────────────────────────────────────────────────────
Manifest:
  Active documents: 50
  Deleted documents: 3
  Parsed: 50
  Chunked: 50
  Upserted: 50

Vector DB:
  Total chunks: 523
  Unique documents: 50
  Namespaces: bge-m3@1024@v1
```

### Logs

```bash
# View indexer logs
tail -f logs/indexer.log

# View errors only
grep ERROR logs/indexer.log
```

---

## 🔧 Advanced Usage

### Python API

```python
from indexer import (
    IncrementalIndexer,
    DocumentParser,
    DocumentChunker,
    Embedder,
    VectorDBUpserter,
    ManifestDB
)

# Full control
manifest = ManifestDB()
parser = DocumentParser()
chunker = DocumentChunker(chunk_size=700, overlap=0.12)
embedder = Embedder(model_namespace="bge-m3@1024@v1")
upserter = VectorDBUpserter(collection_name="papers_v1")

# Process file
parsed = parser.parse_file('paper.pdf')
chunks = chunker.chunk_document(parsed)
embeddings = embedder.embed_chunks(chunks)
upserter.upsert_chunks(chunks, embeddings, "bge-m3@1024@v1")
```

### Watch for Changes

```bash
# Auto-refresh on file changes
make watch
```

Requires `fswatch` (macOS) or `inotifywait` (Linux):
```bash
# macOS
brew install fswatch

# Linux
apt-get install inotify-tools
```

---

## 🐛 Troubleshooting

### Issue: "Document not reprocessing"

**Cause:** SHA256 hash unchanged

**Solution:**
```bash
# Force reindex
make reindex

# Or delete from manifest and re-add
python -c "
from indexer.metadata import ManifestDB
manifest = ManifestDB()
manifest.mark_deleted('doc_id', soft=False)
"
make index
```

### Issue: "PyMuPDF not found"

**Cause:** Missing PDF parsing library

**Solution:**
```bash
pip install pymupdf
```

### Issue: "Smoke test failing"

**Cause:** Index quality below threshold

**Check:**
1. Are papers actually indexed?
```bash
make stats
```

2. Is retrieval working?
```python
from rag.pipeline import RAGPipeline
pipeline = RAGPipeline()
pipeline.initialize()
result = pipeline.query("What is Transformer?", top_k=10)
print(f"Retrieved: {result['all_retrieved']}")
```

3. Check manifest consistency
```python
from indexer.metadata import ManifestDB
manifest = ManifestDB()
stats = manifest.get_stats()
print(stats)
```

---

## 🎯 Best Practices

### 1. Regular Incremental Updates
```bash
# Daily cron
0 2 * * * cd /path/to/academic-rag-system && make index
```

### 2. Namespace Versioning
```
bge-m3@1024@v1  # Production
bge-m3@1024@v2  # Staging/testing
bge-m3@1024@v3  # Development
```

### 3. Smoke Tests in CI
```yaml
# .github/workflows/index.yml
- name: Index papers
  run: make index

- name: Smoke test
  run: make smoke
```

### 4. Backup Manifest
```bash
# Backup before major changes
cp data/manifest.db data/manifest.db.backup
```

---

## 📈 Performance

### Indexing Speed

**Single document (10 pages):**
- Parse: ~1s
- Chunk: ~0.1s
- Embed (700 tokens): ~0.5s (GPU) / ~2s (CPU)
- Upsert: ~0.1s

**Total:** ~2s per document (GPU) / ~3.5s (CPU)

**50 documents:** ~2-3 minutes (GPU) / ~5-7 minutes (CPU)

### Optimization Tips

1. **Batch size**: Increase for GPU
```yaml
embedding:
  batch_size: 64  # GPU
  # batch_size: 16  # CPU
```

2. **Parallel workers**: Process multiple files
```yaml
processing:
  workers: 4
```

3. **Skip processed**: Don't reprocess unchanged
```yaml
processing:
  skip_processed: true
```

---

## 🔗 Integration with RAG Pipeline

The indexer automatically creates/updates collections that the RAG pipeline queries:

```python
# RAG pipeline automatically uses latest indexed collection
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.initialize()

# Queries against indexed documents
result = pipeline.query("What is Transformer Architecture?")
```

**Backwards Compatible:** Existing query API unchanged!

---

## 📝 Summary

✅ **Incremental**: Only processes new/modified documents
✅ **Idempotent**: Safe to run multiple times
✅ **Versioned**: Namespace-based model migration
✅ **Validated**: Smoke tests ensure quality
✅ **Automated**: Make targets and scripts
✅ **Monitored**: Statistics and logging

**Result:** Production-ready incremental indexing system!

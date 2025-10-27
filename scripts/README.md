# Scripts

Utility scripts for data collection, processing, and evaluation.

## 📂 Directory Structure

```
scripts/
├── data_collection/    # Collect papers from arXiv
├── processing/         # Process PDFs and build indices
├── evaluation/         # Evaluate system performance
└── legacy/            # Deprecated scripts (use app/cli.py instead)
```

## 🔨 Data Collection

Collect academic papers from arXiv:

```bash
# Collect classic papers
python scripts/data_collection/collect_classic_papers.py

# Download PDFs
python scripts/data_collection/download_pdfs.py
```

## ⚙️ Processing

Process PDFs and build vector indices:

```bash
# Process PDFs and extract content
python scripts/processing/process_pdf_fulltext.py

# Or use Makefile
make index
```

## 📊 Evaluation

Evaluate system performance:

```bash
# Run evaluation
python scripts/evaluation/evaluate_rag_system.py

# Or use Makefile
make evaluate
```

## 🗑️ Legacy Scripts

The `legacy/` directory contains deprecated scripts that have been replaced by the new `app/cli.py` interface.

**Instead of**:
```bash
python scripts/legacy/main_rag_system.py
```

**Use**:
```bash
python -m app.cli query --interactive
# or
make run
```

---

**See also**: [Makefile](../Makefile) for all available commands

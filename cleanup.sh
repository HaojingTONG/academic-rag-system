#!/bin/bash
# Project Cleanup Script - 自动整理项目结构
# =============================================

set -e

echo "🧹 Starting project cleanup..."
echo ""

# 创建目录
echo "📁 Creating directories..."
mkdir -p docs/{guides,features,refactoring,api,architecture}
mkdir -p scripts/{data_collection,processing,evaluation,legacy}
mkdir -p examples/{demos,notebooks}
mkdir -p tests/legacy

echo "✓ Directories created"
echo ""

# 移动文档
echo "📖 Moving documentation files..."

# 重构文档
for file in REFACTORING_PLAN.md REFACTORING_EXECUTIVE_SUMMARY.md \
            PHASE2_COMPLETION_REPORT.md FINAL_REFACTORING_REPORT.md \
            MIGRATION.md PROJECT_CLEANUP_PLAN.md; do
    if [ -f "$file" ]; then
        mv "$file" docs/refactoring/
        echo "  ✓ $file → docs/refactoring/"
    fi
done

# 用户指南
if [ -f "QUICKSTART.md" ]; then
    mv QUICKSTART.md docs/guides/quickstart.md
    echo "  ✓ QUICKSTART.md → docs/guides/quickstart.md"
fi
if [ -f "QUICK_REFERENCE.md" ]; then
    mv QUICK_REFERENCE.md docs/guides/quick_reference.md
    echo "  ✓ QUICK_REFERENCE.md → docs/guides/quick_reference.md"
fi
if [ -f "USAGE_EXAMPLES.md" ]; then
    mv USAGE_EXAMPLES.md docs/guides/usage_examples.md
    echo "  ✓ USAGE_EXAMPLES.md → docs/guides/usage_examples.md"
fi

# 功能说明
for file in EVALUATION_FRAMEWORK.md FIGURE_PRESERVATION_SUMMARY.md \
            TABLE_PRESERVATION_SUMMARY.md column_layout_summary.md \
            fine_grained_features_summary.md; do
    if [ -f "$file" ]; then
        # 转换为小写下划线命名
        newname=$(echo "$file" | tr '[:upper:]' '[:lower:]' | sed 's/_summary//')
        mv "$file" "docs/features/$newname"
        echo "  ✓ $file → docs/features/$newname"
    fi
done

echo ""

# 移动Python脚本
echo "🔨 Moving Python scripts..."

# 数据收集
for file in collect_classic_papers.py collect_high_quality_papers.py \
            download_pdfs.py integrate_papers.py; do
    if [ -f "$file" ]; then
        mv "$file" scripts/data_collection/
        echo "  ✓ $file → scripts/data_collection/"
    fi
done

# 处理脚本
for file in process_pdf_fulltext.py improved_text_cleaner.py \
            short_section_handler.py; do
    if [ -f "$file" ]; then
        mv "$file" scripts/processing/
        echo "  ✓ $file → scripts/processing/"
    fi
done

# 评估脚本
if [ -f "evaluate_rag_system.py" ]; then
    mv evaluate_rag_system.py scripts/evaluation/
    echo "  ✓ evaluate_rag_system.py → scripts/evaluation/"
fi

# 主脚本（已过时）
if [ -f "main_rag_system.py" ]; then
    mv main_rag_system.py scripts/legacy/
    echo "  ✓ main_rag_system.py → scripts/legacy/ (deprecated)"
fi

echo ""

# 移动示例
echo "💡 Moving examples..."
for file in demo_*.py; do
    if [ -f "$file" ]; then
        mv "$file" examples/demos/
        echo "  ✓ $file → examples/demos/"
    fi
done

echo ""

# 移动旧测试
echo "🧪 Moving legacy tests..."
for file in test_*.py; do
    if [ -f "$file" ]; then
        mv "$file" tests/legacy/
        echo "  ✓ $file → tests/legacy/"
    fi
done

echo ""

# 创建索引文件
echo "📝 Creating index files..."

# docs/README.md
cat > docs/README.md << 'EOF'
# Documentation Index

Welcome to the Academic RAG System documentation!

## 📚 User Guides

Get started and learn how to use the system:

- **[Quickstart Guide](guides/quickstart.md)** - Get started in 5 minutes
- **[Quick Reference](guides/quick_reference.md)** - Common commands and usage
- **[Usage Examples](guides/usage_examples.md)** - Real-world examples

## ✨ Features

Learn about advanced features:

- **[Evaluation Framework](features/evaluation_framework.md)** - How we evaluate RAG quality
- **[Figure Preservation](features/figure_preservation.md)** - Extracting figures from PDFs
- **[Table Preservation](features/table_preservation.md)** - Extracting tables from PDFs
- **[Column Layout](features/column_layout.md)** - Handling multi-column PDFs
- **[Fine-grained Features](features/fine_grained_features.md)** - Advanced chunking features

## 🔧 Refactoring Documentation

Technical documentation about the v2.0 refactoring:

- **[Refactoring Plan](refactoring/REFACTORING_PLAN.md)** - Complete refactoring plan
- **[Migration Guide](refactoring/MIGRATION.md)** - How to migrate from v1.0 to v2.0
- **[Phase 2 Report](refactoring/PHASE2_COMPLETION_REPORT.md)** - Module consolidation
- **[Final Report](refactoring/FINAL_REFACTORING_REPORT.md)** - Complete project summary
- **[Cleanup Plan](refactoring/PROJECT_CLEANUP_PLAN.md)** - Project organization

## 📖 API Reference

- API documentation coming soon...

## 🏗️ Architecture

- Architecture documentation coming soon...

## 🔗 Quick Links

- [Main README](../README.md) - Project overview
- [GitHub Repository](https://github.com/your-org/academic-rag-system) - Source code
- [Issue Tracker](https://github.com/your-org/academic-rag-system/issues) - Report bugs

---

**Last Updated**: 2024-01-15
EOF

echo "  ✓ docs/README.md created"

# examples/README.md
cat > examples/README.md << 'EOF'
# Examples

This directory contains example code and demonstrations.

## 🎯 Demos

Demonstrations of specific features:

- **[demo_figure_preservation.py](demos/demo_figure_preservation.py)** - Figure extraction demo
- **[demo_table_preservation.py](demos/demo_table_preservation.py)** - Table extraction demo

## 📓 Notebooks

Jupyter notebooks for interactive exploration:

Coming soon...

## 💡 Usage

### Running Demos

```bash
# Figure preservation
python examples/demos/demo_figure_preservation.py

# Table preservation
python examples/demos/demo_table_preservation.py
```

### Creating Notebooks

```bash
jupyter notebook examples/notebooks/
```

---

**See also**: [User Guides](../docs/guides/) for more examples
EOF

echo "  ✓ examples/README.md created"

# scripts/README.md
cat > scripts/README.md << 'EOF'
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
EOF

echo "  ✓ scripts/README.md created"

echo ""

# 显示结果
echo "✅ Cleanup complete!"
echo ""
echo "📊 Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Root directory now has: $(ls -1 | wc -l | tr -d ' ') items"
echo ""
echo "📁 New structure:"
ls -1 | grep -v "^\." | head -15
echo ""
echo "📖 Documentation: $(find docs -type f -name "*.md" | wc -l | tr -d ' ') files"
echo "🔨 Scripts: $(find scripts -type f -name "*.py" | wc -l | tr -d ' ') files"
echo "💡 Examples: $(find examples -type f -name "*.py" | wc -l | tr -d ' ') files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🎯 Next steps:"
echo "  1. Review changes: tree -L 2 -I '__pycache__|*.pyc|venv*'"
echo "  2. Update imports if needed"
echo "  3. Test system: make smoke"
echo "  4. Commit changes: git add . && git commit -m 'Clean up project structure'"
echo ""
echo "📝 See PROJECT_CLEANUP_PLAN.md for details (now in docs/refactoring/)"

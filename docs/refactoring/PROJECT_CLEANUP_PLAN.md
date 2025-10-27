# Project Cleanup Plan - 项目整理方案 🧹

> **目标**: 让项目结构清爽、专业、易维护
> **原则**: 文档归档、脚本分类、测试规范、根目录简洁

---

## 📊 当前问题

**根目录混乱**：31个文件散落
- 14个 Markdown 文档
- 17个 Python 脚本
- 大量过时/重复文件

---

## 🎯 目标结构

### 理想的根目录（只保留8-10个核心文件）

```
academic-rag-system/
├── README.md                 ⭐ 项目主文档
├── requirements.txt          📦 依赖列表
├── requirements-dev.txt      🔧 开发依赖
├── Makefile                  🛠️ 构建命令
├── .env.example              ⚙️ 环境变量模板
├── .gitignore                🚫 Git忽略
├── pyproject.toml            📝 项目配置（可选）
├── setup.py                  📦 安装脚本（可选）
│
├── rag/                      📚 核心RAG模块
├── app/                      🚀 应用入口
├── configs/                  ⚙️ 配置文件
├── tests/                    🧪 测试套件
├── scripts/                  🔨 工具脚本
├── docs/                     📖 所有文档
├── examples/                 💡 示例代码
├── data/                     💾 数据文件
└── logs/                     📋 日志（gitignore）
```

---

## 📂 文件重组方案

### 1. Markdown 文档整理

#### **保留在根目录**（仅1个）
```
README.md  ⭐ - 项目主文档（需重写）
```

#### **移动到 docs/**

##### docs/refactoring/（重构相关文档）
```
REFACTORING_PLAN.md                  → docs/refactoring/
REFACTORING_EXECUTIVE_SUMMARY.md     → docs/refactoring/
PHASE2_COMPLETION_REPORT.md          → docs/refactoring/
FINAL_REFACTORING_REPORT.md          → docs/refactoring/
MIGRATION.md                         → docs/refactoring/
```

##### docs/guides/（用户指南）
```
QUICKSTART.md                        → docs/guides/
QUICK_REFERENCE.md                   → docs/guides/
USAGE_EXAMPLES.md                    → docs/guides/
```

##### docs/features/（功能说明）
```
EVALUATION_FRAMEWORK.md              → docs/features/
FIGURE_PRESERVATION_SUMMARY.md       → docs/features/
TABLE_PRESERVATION_SUMMARY.md        → docs/features/
column_layout_summary.md             → docs/features/
fine_grained_features_summary.md     → docs/features/
```

---

### 2. Python 脚本整理

#### **数据收集脚本** → `scripts/data_collection/`
```
collect_classic_papers.py            → scripts/data_collection/
collect_high_quality_papers.py       → scripts/data_collection/
download_pdfs.py                     → scripts/data_collection/
integrate_papers.py                  → scripts/data_collection/
```

#### **处理脚本** → `scripts/processing/`
```
process_pdf_fulltext.py              → scripts/processing/
improved_text_cleaner.py             → scripts/processing/
short_section_handler.py             → scripts/processing/
```

#### **评估脚本** → `scripts/evaluation/`
```
evaluate_rag_system.py               → scripts/evaluation/
```

#### **主脚本** → `scripts/legacy/`（或删除）
```
main_rag_system.py                   → scripts/legacy/main_rag_system.py
                                     （已被 app/cli.py 替代）
```

#### **Demo 示例** → `examples/demos/`
```
demo_figure_preservation.py          → examples/demos/
demo_table_preservation.py           → examples/demos/
```

#### **测试文件** → `tests/legacy/`（或删除）
```
test_column_layout.py                → tests/legacy/
test_figure_preservation.py          → tests/legacy/
test_table_preservation.py           → tests/legacy/
test_fine_grained_features.py        → tests/legacy/
test_improved_chunking.py            → tests/legacy/
test_short_section_processing.py     → tests/legacy/
```

---

## 🗂️ 新的目录结构

### 完整结构预览

```
academic-rag-system/
│
├── README.md                         ⭐ 唯一根目录文档
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── .env.example
├── .gitignore
│
├── rag/                              📚 核心模块
│   ├── __init__.py
│   ├── retriever.py
│   ├── ranker.py
│   └── pipeline.py
│
├── app/                              🚀 应用入口
│   ├── __init__.py
│   ├── main.py                       (FastAPI)
│   └── cli.py                        (CLI)
│
├── configs/                          ⚙️ 配置
│   ├── config.yaml
│   ├── config_loader.py
│   └── __init__.py
│
├── tests/                            🧪 测试
│   ├── pytest.ini
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── legacy/                       (旧测试脚本)
│       ├── test_column_layout.py
│       ├── test_figure_preservation.py
│       └── ...
│
├── scripts/                          🔨 工具脚本
│   ├── dev_bootstrap.sh
│   ├── run_smoke.sh
│   ├── data_collection/              (数据收集)
│   │   ├── collect_classic_papers.py
│   │   ├── download_pdfs.py
│   │   └── ...
│   ├── processing/                   (数据处理)
│   │   ├── process_pdf_fulltext.py
│   │   └── ...
│   ├── evaluation/                   (评估)
│   │   └── evaluate_rag_system.py
│   └── legacy/                       (过时脚本)
│       └── main_rag_system.py
│
├── examples/                         💡 示例代码
│   ├── README.md                     (示例说明)
│   ├── demos/                        (功能演示)
│   │   ├── demo_figure_preservation.py
│   │   └── demo_table_preservation.py
│   └── notebooks/                    (Jupyter notebooks)
│       └── rag_tutorial.ipynb
│
├── docs/                             📖 所有文档
│   ├── README.md                     (文档索引)
│   ├── guides/                       (用户指南)
│   │   ├── quickstart.md
│   │   ├── quick_reference.md
│   │   └── usage_examples.md
│   ├── features/                     (功能说明)
│   │   ├── evaluation_framework.md
│   │   ├── figure_preservation.md
│   │   └── table_preservation.md
│   ├── refactoring/                  (重构文档)
│   │   ├── refactoring_plan.md
│   │   ├── migration.md
│   │   └── completion_reports.md
│   ├── api/                          (API文档)
│   │   └── api_reference.md
│   └── architecture/                 (架构文档)
│       └── system_design.md
│
├── data/                             💾 数据（gitignore大文件）
├── vector_db/                        🔍 向量数据库（gitignore）
├── logs/                             📋 日志（gitignore）
└── .backup/                          💾 备份（gitignore）
```

---

## 🛠️ 执行方案（分步骤）

### Step 1: 创建新目录

```bash
# 创建文档目录
mkdir -p docs/{guides,features,refactoring,api,architecture}

# 创建脚本目录
mkdir -p scripts/{data_collection,processing,evaluation,legacy}

# 创建示例目录
mkdir -p examples/{demos,notebooks}

# 创建测试目录（如果不存在）
mkdir -p tests/legacy
```

### Step 2: 移动文档文件

```bash
# 重构文档
mv REFACTORING_PLAN.md docs/refactoring/
mv REFACTORING_EXECUTIVE_SUMMARY.md docs/refactoring/
mv PHASE2_COMPLETION_REPORT.md docs/refactoring/
mv FINAL_REFACTORING_REPORT.md docs/refactoring/
mv MIGRATION.md docs/refactoring/

# 用户指南
mv QUICKSTART.md docs/guides/quickstart.md
mv QUICK_REFERENCE.md docs/guides/quick_reference.md
mv USAGE_EXAMPLES.md docs/guides/usage_examples.md

# 功能说明
mv EVALUATION_FRAMEWORK.md docs/features/evaluation_framework.md
mv FIGURE_PRESERVATION_SUMMARY.md docs/features/figure_preservation.md
mv TABLE_PRESERVATION_SUMMARY.md docs/features/table_preservation.md
mv column_layout_summary.md docs/features/column_layout.md
mv fine_grained_features_summary.md docs/features/fine_grained_features.md
```

### Step 3: 移动Python脚本

```bash
# 数据收集脚本
mv collect_classic_papers.py scripts/data_collection/
mv collect_high_quality_papers.py scripts/data_collection/
mv download_pdfs.py scripts/data_collection/
mv integrate_papers.py scripts/data_collection/

# 处理脚本
mv process_pdf_fulltext.py scripts/processing/
mv improved_text_cleaner.py scripts/processing/
mv short_section_handler.py scripts/processing/

# 评估脚本
mv evaluate_rag_system.py scripts/evaluation/

# 主脚本（已过时）
mv main_rag_system.py scripts/legacy/

# Demo 示例
mv demo_figure_preservation.py examples/demos/
mv demo_table_preservation.py examples/demos/

# 旧测试
mv test_column_layout.py tests/legacy/
mv test_figure_preservation.py tests/legacy/
mv test_table_preservation.py tests/legacy/
mv test_fine_grained_features.py tests/legacy/
mv test_improved_chunking.py tests/legacy/
mv test_short_section_processing.py tests/legacy/
```

### Step 4: 创建索引文件

```bash
# docs/README.md
cat > docs/README.md << 'EOF'
# Documentation Index

## 📚 User Guides
- [Quickstart Guide](guides/quickstart.md) - Get started in 5 minutes
- [Quick Reference](guides/quick_reference.md) - Common commands
- [Usage Examples](guides/usage_examples.md) - Code examples

## ✨ Features
- [Evaluation Framework](features/evaluation_framework.md)
- [Figure Preservation](features/figure_preservation.md)
- [Table Preservation](features/table_preservation.md)

## 🔧 Refactoring
- [Refactoring Plan](refactoring/REFACTORING_PLAN.md)
- [Migration Guide](refactoring/MIGRATION.md)
- [Completion Reports](refactoring/)

## 📖 API Reference
- [API Documentation](api/api_reference.md)
EOF

# examples/README.md
cat > examples/README.md << 'EOF'
# Examples

## Demos
- [Figure Preservation](demos/demo_figure_preservation.py)
- [Table Preservation](demos/demo_table_preservation.py)

## Notebooks
Coming soon...
EOF
```

### Step 5: 更新 README.md

创建一个清爽的根目录 README。

---

## ✅ 清理检查清单

执行完后检查：

```bash
# 1. 根目录应该只有这些文件
ls -1 | grep -E "^[^.]" | sort
# 预期输出:
# Makefile
# README.md
# app/
# configs/
# data/
# docs/
# examples/
# logs/
# rag/
# requirements-dev.txt
# requirements.txt
# scripts/
# src/
# tests/
# vector_db/

# 2. 文档都在 docs/
ls -R docs/

# 3. 脚本都在 scripts/
ls -R scripts/

# 4. 示例都在 examples/
ls -R examples/
```

---

## 🎨 额外优化建议

### 1. 重命名文件（小写+下划线）

```bash
# 统一命名风格（全部小写，下划线分隔）
cd docs/refactoring/
rename 's/([A-Z])/_\l$1/g' *.md  # Linux
# 或手动重命名为:
# REFACTORING_PLAN.md → refactoring_plan.md
```

### 2. 创建 .gitignore 条目

```gitignore
# 添加到 .gitignore
logs/
*.log
__pycache__/
*.pyc
.DS_Store
.env
vector_db/
data/raw_papers/
data/embedding_cache/
.backup/
```

### 3. 创建 Makefile 命令

```makefile
# 添加到 Makefile
clean-docs:  ## Clean documentation build artifacts
	rm -rf docs/_build

reorganize:  ## Show directory structure
	tree -L 2 -I '__pycache__|*.pyc|venv*'

validate:  ## Validate project structure
	@echo "Checking root directory..."
	@ls -1 | wc -l
	@echo "Should be around 15 items"
```

---

## 📋 执行优先级

### 高优先级（立即执行）
1. ✅ 移动文档到 docs/
2. ✅ 移动脚本到 scripts/
3. ✅ 移动测试到 tests/legacy/
4. ✅ 移动 demo 到 examples/

### 中优先级（本周完成）
1. ⚠️ 创建索引文件（README.md）
2. ⚠️ 重写根目录 README.md
3. ⚠️ 更新 .gitignore

### 低优先级（有时间再做）
1. 📝 文件重命名（统一命名风格）
2. 📝 删除完全过时的文件
3. 📝 创建 examples/notebooks/

---

## 🚀 一键执行脚本

创建 `cleanup.sh` 自动执行：

```bash
#!/bin/bash
# cleanup.sh - 自动整理项目结构

set -e

echo "🧹 Starting project cleanup..."

# 创建目录
echo "📁 Creating directories..."
mkdir -p docs/{guides,features,refactoring,api,architecture}
mkdir -p scripts/{data_collection,processing,evaluation,legacy}
mkdir -p examples/{demos,notebooks}
mkdir -p tests/legacy

# 移动文档
echo "📖 Moving documentation..."
mv REFACTORING_*.md docs/refactoring/ 2>/dev/null || true
mv PHASE2_*.md docs/refactoring/ 2>/dev/null || true
mv FINAL_*.md docs/refactoring/ 2>/dev/null || true
mv MIGRATION.md docs/refactoring/ 2>/dev/null || true
mv QUICKSTART.md docs/guides/quickstart.md 2>/dev/null || true
mv QUICK_REFERENCE.md docs/guides/quick_reference.md 2>/dev/null || true
mv USAGE_EXAMPLES.md docs/guides/usage_examples.md 2>/dev/null || true
mv *_SUMMARY.md docs/features/ 2>/dev/null || true
mv *_summary.md docs/features/ 2>/dev/null || true

# 移动脚本
echo "🔨 Moving scripts..."
mv collect_*.py scripts/data_collection/ 2>/dev/null || true
mv download_*.py scripts/data_collection/ 2>/dev/null || true
mv integrate_*.py scripts/data_collection/ 2>/dev/null || true
mv process_*.py scripts/processing/ 2>/dev/null || true
mv improved_*.py scripts/processing/ 2>/dev/null || true
mv short_section_*.py scripts/processing/ 2>/dev/null || true
mv evaluate_*.py scripts/evaluation/ 2>/dev/null || true
mv main_rag_system.py scripts/legacy/ 2>/dev/null || true

# 移动示例
echo "💡 Moving examples..."
mv demo_*.py examples/demos/ 2>/dev/null || true

# 移动测试
echo "🧪 Moving legacy tests..."
mv test_*.py tests/legacy/ 2>/dev/null || true

echo "✅ Cleanup complete!"
echo ""
echo "📊 Root directory now has:"
ls -1 | grep -v "^\." | wc -l
echo "files/directories"
echo ""
echo "Run 'tree -L 2 -I __pycache__' to see new structure"
```

---

## 📊 预期效果

### Before（现在）
```
根目录: 31+ 个文件
- 混乱、难找
- 无组织
- 不专业
```

### After（整理后）
```
根目录: ~15 个项目
- 清爽、专业
- 分类清晰
- 易于维护
```

---

**执行建议**: 先运行一键脚本，然后手动检查和调整！

---

**Document Version**: 1.0
**Status**: Ready to Execute
**Estimated Time**: 15 minutes

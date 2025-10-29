#!/bin/bash
# Project Structure Refactoring Script
# ======================================
# 自动执行项目结构优化的第一阶段
#
# 使用方法：
#   chmod +x refactor.sh
#   ./refactor.sh

set -e  # 遇到错误立即退出

echo "🚀 开始项目结构优化..."
echo ""

# 检查是否在正确的目录
if [ ! -f "README.md" ] || [ ! -d "rag" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    exit 1
fi

# 创建备份
echo "📦 创建备份..."
BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p .backup/$BACKUP_NAME
git stash push -m "Before refactoring - $BACKUP_NAME" 2>/dev/null || true
echo "✅ 备份完成（使用 git stash 保存）"
echo ""

# 阶段1: 创建目标目录
echo "📁 创建目标目录结构..."
mkdir -p docs/summaries
mkdir -p docs/setup
mkdir -p docs/guides
mkdir -p scripts
mkdir -p tests/results
echo "✅ 目录创建完成"
echo ""

# 阶段2: 移动文档文件
echo "📄 整理文档文件..."
MOVED_DOCS=0

# 移动SUMMARY类文档
for file in *_SUMMARY.md *_IMPROVEMENTS.md *_OPTIMIZATION.md FRONTEND_*.md GENERATOR_*.md RAG_*.md ANSWER_*.md; do
    if [ -f "$file" ]; then
        mv "$file" docs/summaries/ 2>/dev/null && ((MOVED_DOCS++)) || true
    fi
done

# 移动设置指南
for file in QUICK_*.md OPENAI_SETUP.md INDEXER_README.md; do
    if [ -f "$file" ]; then
        mv "$file" docs/setup/ 2>/dev/null && ((MOVED_DOCS++)) || true
    fi
done

echo "✅ 移动了 $MOVED_DOCS 个文档文件"
echo ""

# 阶段3: 移动测试文件
echo "🧪 整理测试文件..."
MOVED_TESTS=0

for file in test_*.py; do
    if [ -f "$file" ]; then
        # 检查文件是否已经在tests/目录
        if [[ "$file" != tests/* ]]; then
            mv "$file" tests/ 2>/dev/null && ((MOVED_TESTS++)) || true
        fi
    fi
done

# 移动测试结果文件
for file in *_results.json citation_test_results.json eval_baseline_results.json; do
    if [ -f "$file" ]; then
        mv "$file" tests/results/ 2>/dev/null || true
    fi
done

echo "✅ 移动了 $MOVED_TESTS 个测试文件"
echo ""

# 阶段4: 移动脚本文件
echo "🔧 整理脚本文件..."
MOVED_SCRIPTS=0

for file in build_bm25_index.py evaluate_rag.py diagnose_system.py verify_fixes.py; do
    if [ -f "$file" ]; then
        mv "$file" scripts/ 2>/dev/null && ((MOVED_SCRIPTS++)) || true
    fi
done

echo "✅ 移动了 $MOVED_SCRIPTS 个脚本文件"
echo ""

# 阶段5: 更新Makefile中的路径
if [ -f "Makefile" ]; then
    echo "📝 更新Makefile路径..."

    # 备份原始Makefile
    cp Makefile Makefile.backup

    # 更新脚本路径
    sed -i.bak 's|python build_bm25_index.py|python scripts/build_bm25_index.py|g' Makefile
    sed -i.bak 's|python evaluate_rag.py|python scripts/evaluate_rag.py|g' Makefile
    sed -i.bak 's|python diagnose_system.py|python scripts/diagnose_system.py|g' Makefile
    sed -i.bak 's|python verify_fixes.py|python scripts/verify_fixes.py|g' Makefile

    rm Makefile.bak 2>/dev/null || true

    echo "✅ Makefile已更新"
    echo ""
fi

# 显示统计信息
echo "📊 清理统计："
echo "   - 移动文档: $MOVED_DOCS 个"
echo "   - 移动测试: $MOVED_TESTS 个"
echo "   - 移动脚本: $MOVED_SCRIPTS 个"
echo ""

# 显示根目录文件数量
ROOT_FILES=$(ls -1 *.md *.py 2>/dev/null | wc -l)
echo "📁 根目录剩余 MD/PY 文件: $ROOT_FILES 个"
echo ""

# 验证关键文件存在
echo "🔍 验证项目结构..."
ISSUES=0

if [ ! -f "README.md" ]; then
    echo "   ⚠️  缺少 README.md"
    ((ISSUES++))
fi

if [ ! -f "requirements.txt" ]; then
    echo "   ⚠️  缺少 requirements.txt"
    ((ISSUES++))
fi

if [ ! -d "rag" ]; then
    echo "   ⚠️  缺少 rag/ 目录"
    ((ISSUES++))
fi

if [ ! -d "app" ]; then
    echo "   ⚠️  缺少 app/ 目录"
    ((ISSUES++))
fi

if [ $ISSUES -eq 0 ]; then
    echo "✅ 项目结构验证通过"
else
    echo "⚠️  发现 $ISSUES 个问题"
fi
echo ""

# 显示下一步建议
echo "✅ 项目结构优化完成！"
echo ""
echo "📋 下一步建议："
echo "   1. 查看变更: git status"
echo "   2. 测试系统: make smoke"
echo "   3. 运行测试: pytest tests/"
echo "   4. 查看优化方案: cat PROJECT_STRUCTURE_OPTIMIZATION.md"
echo ""
echo "💡 如需回滚："
echo "   git stash pop  # 恢复到优化前状态"
echo ""

# 提示查看新文件
if [ -f "PROJECT_STRUCTURE_OPTIMIZATION.md" ]; then
    echo "📖 阅读完整优化方案:"
    echo "   cat PROJECT_STRUCTURE_OPTIMIZATION.md"
    echo ""
fi

echo "🎉 完成！"

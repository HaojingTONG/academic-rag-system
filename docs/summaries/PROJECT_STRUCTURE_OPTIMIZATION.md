# 📁 项目结构优化方案

## 🎯 优化目标

1. **清理根目录** - 减少根目录文件数量，提高可维护性
2. **消除重复** - 合并功能重复的模块
3. **统一管理** - 配置、文档、测试统一组织
4. **生产就绪** - 添加Docker、CI/CD等生产配置
5. **标准化** - 遵循Python项目最佳实践

## 📊 当前问题分析

### 问题1：根目录过于混乱 ⚠️

**现状：**
```
根目录有21个文件：
✗ test_*.py (6个)           → 应该在tests/
✗ *_SUMMARY.md (9个)        → 应该在docs/
✗ build_bm25_index.py       → 应该在scripts/
✗ evaluate_rag.py           → 应该在scripts/
```

**影响：**
- 难以快速找到核心文件
- 新人上手困难
- 不符合Python项目规范

### 问题2：模块功能重复 ⚠️

**现状：**
```
rag/               (新模块，统一API)
├── retriever.py
├── generator.py
└── ...

src/               (旧模块，分散功能)
├── retriever/     ⚠️ 与rag/retriever.py重复
├── generator/     ⚠️ 与rag/generator.py重复
└── embedding/     ⚠️ 部分功能重复
```

**影响：**
- 维护成本高（两套代码）
- 容易出现不一致
- 新功能不知道加在哪里

### 问题3：配置分散 ⚠️

**现状：**
```
configs/config.yaml          (主配置)
configs/retrieval.yaml       (检索配置)
configs/index.yaml          (索引配置)
.env                        (环境变量)
rag/prompt_templates.yaml   (提示模板)
```

**影响：**
- 配置不好管理
- 容易遗漏配置项
- 环境切换困难

### 问题4：缺少生产配置 ⚠️

**缺失：**
- ✗ Dockerfile
- ✗ docker-compose.yml
- ✗ CI/CD配置
- ✗ 健康检查
- ✗ 监控配置

## 🚀 优化方案

### 阶段1：根目录清理（立即执行）

#### 1.1 移动测试文件

```bash
# 移动根目录测试文件到tests/
mv test_*.py tests/

# 更新imports（如果需要）
# 在tests/中的文件添加：
# import sys; sys.path.insert(0, '..')
```

#### 1.2 整理文档

```bash
# 创建docs/summaries/
mkdir -p docs/summaries docs/setup

# 移动SUMMARY文档
mv *_SUMMARY.md docs/summaries/
mv *_IMPROVEMENTS.md docs/summaries/

# 移动设置文档
mv QUICK_START.md docs/setup/
mv OPENAI_SETUP.md docs/setup/
mv QUICK_BACKEND_START.md docs/setup/
mv INDEXER_README.md docs/setup/
```

#### 1.3 整理脚本

```bash
# 移动脚本文件
mv build_bm25_index.py scripts/
mv evaluate_rag.py scripts/
mv diagnose_system.py scripts/
mv verify_fixes.py scripts/

# 更新Makefile中的路径
# 将 build_bm25_index.py 改为 scripts/build_bm25_index.py
```

#### 1.4 清理后的根目录

```
academic-rag-system/
├── README.md              ✅ 项目主文档
├── .env.example          ✅ 配置示例
├── .gitignore            ✅ Git配置
├── requirements.txt      ✅ 依赖
├── requirements-dev.txt  ✅ 开发依赖
├── Makefile              ✅ 任务管理
│
├── app/                  ✅ FastAPI应用
├── rag/                  ✅ RAG核心模块
├── indexer/              ✅ 索引器
├── configs/              ✅ 配置
├── docs/                 ✅ 文档
├── scripts/              ✅ 脚本
├── tests/                ✅ 测试
├── frontend/             ✅ 前端
├── data/                 ✅ 数据
└── vector_db/            ✅ 向量数据库
```

### 阶段2：模块重组（短期）

#### 2.1 决策：保留rag/还是src/

**推荐方案：完全使用rag/模块** ✅

**原因：**
1. rag/是新设计的统一API
2. 代码更清晰，文档更好
3. 已经在app/main.py中使用
4. 更符合现代Python项目结构

**执行计划：**

```bash
# 步骤1: 识别src/中仍在使用的模块
grep -r "from src" . --include="*.py" | grep -v ".venv" | grep -v "vector_db"

# 步骤2: 迁移必要模块到rag/
# src/embedding/ → rag/models/embedding.py
# src/generator/llm_client.py → rag/models/llm_client.py

# 步骤3: 更新所有imports
# 从: from src.embedding import EmbeddingManager
# 到: from rag.models.embedding import EmbeddingManager

# 步骤4: 归档旧代码
mv src .archive/src_deprecated_$(date +%Y%m%d)
```

#### 2.2 建议的rag/目录结构

```bash
rag/
├── __init__.py                 # 导出主要类
├── pipeline.py                 # RAG主管道
│
├── retrieval/                  # 检索模块
│   ├── __init__.py
│   ├── retriever.py           # 向量+BM25检索
│   ├── query_analyzer.py      # 查询分析
│   └── vector_store.py        # 向量存储
│
├── ranking/                    # 排序模块
│   ├── __init__.py
│   ├── ranker.py              # 综合排序
│   ├── reranker.py            # 重排序
│   ├── diversifier.py         # 多样化
│   └── filter.py              # 过滤
│
├── generation/                 # 生成模块
│   ├── __init__.py
│   ├── generator.py           # 答案生成
│   ├── composer.py            # 提示组合
│   ├── templates.py           # 模板管理
│   └── quality.py             # 质量检查
│
├── models/                     # 模型相关
│   ├── __init__.py
│   ├── embedding.py           # 嵌入模型
│   └── llm_client.py          # LLM客户端
│
└── utils/                      # 工具函数
    ├── __init__.py
    ├── citation.py            # 引用处理
    └── metrics.py             # 评估指标
```

### 阶段3：配置统一（短期）

#### 3.1 配置层次结构

```yaml
# configs/default.yaml (基础配置)
system:
  name: "Academic RAG System"
  version: "2.0.0"

# configs/development.yaml (开发环境)
extends: default.yaml
system:
  log_level: "DEBUG"
api:
  reload: true

# configs/production.yaml (生产环境)
extends: default.yaml
system:
  log_level: "WARNING"
api:
  workers: 8
```

#### 3.2 环境变量管理

```bash
# .env.example (示例，提交到git)
OPENAI_API_KEY=your_key_here
LLM_BACKEND=openai
ENVIRONMENT=development

# .env (本地，不提交)
OPENAI_API_KEY=sk-proj-xxx...
LLM_BACKEND=openai
ENVIRONMENT=development

# .env.production (生产，不提交)
OPENAI_API_KEY=sk-prod-xxx...
LLM_BACKEND=openai
ENVIRONMENT=production
```

### 阶段4：添加生产配置（中期）

#### 4.1 Docker化

**Dockerfile:**
```dockerfile
# 🆕 Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "app/main.py"]
```

**docker-compose.yml:**
```yaml
# 🆕 docker-compose.yml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ENVIRONMENT=production
    volumes:
      - ./data:/app/data
      - ./vector_db:/app/vector_db
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_RAG_BASE_URL=http://backend:8000
    depends_on:
      - backend

  # 可选：添加监控
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
```

#### 4.2 CI/CD配置

**GitHub Actions:**
```yaml
# 🆕 .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: |
        pytest tests/ -v --cov=rag --cov=app

    - name: Lint
      run: |
        flake8 rag/ app/ --max-line-length=120
        black --check rag/ app/
```

#### 4.3 添加pyproject.toml

```toml
# 🆕 pyproject.toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "academic-rag-system"
version = "2.0.0"
description = "Academic paper RAG system with advanced retrieval and generation"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
requires-python = ">=3.9"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "chromadb>=0.4.0",
    "openai>=1.3.0",
    # ... 其他依赖
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
]

[tool.black]
line-length = 120
target-version = ['py311']

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"

[tool.coverage.run]
source = ["rag", "app", "indexer"]
omit = ["tests/*", "*/migrations/*"]
```

### 阶段5：文档完善（中期）

#### 5.1 文档结构

```bash
docs/
├── README.md                   # 文档索引
│
├── setup/                      # 设置指南
│   ├── installation.md
│   ├── quick_start.md
│   ├── openai_setup.md
│   └── configuration.md
│
├── guides/                     # 使用指南
│   ├── basic_usage.md
│   ├── advanced_features.md
│   ├── api_reference.md
│   └── troubleshooting.md
│
├── architecture/               # 架构文档
│   ├── system_design.md
│   ├── rag_pipeline.md
│   ├── indexer_design.md
│   └── adr/                   # 架构决策记录
│       └── 001_rag_optimization.md
│
├── development/                # 开发文档
│   ├── contributing.md
│   ├── testing.md
│   ├── coding_standards.md
│   └── release_process.md
│
└── summaries/                  # 优化总结
    ├── rag_optimization.md
    ├── generation_optimization.md
    └── frontend_fix.md
```

## 📋 执行检查清单

### 第一周：快速清理 🔥

- [ ] 移动test_*.py到tests/
- [ ] 移动*_SUMMARY.md到docs/summaries/
- [ ] 移动脚本到scripts/
- [ ] 更新Makefile路径
- [ ] 更新.gitignore
- [ ] 测试所有功能是否正常

### 第二周：模块整理

- [ ] 分析src/和rag/的依赖关系
- [ ] 决定保留哪个模块体系
- [ ] 创建迁移计划
- [ ] 执行代码迁移
- [ ] 更新所有imports
- [ ] 运行完整测试套件

### 第三周：配置统一

- [ ] 重组configs/目录
- [ ] 添加环境配置文件
- [ ] 统一配置加载逻辑
- [ ] 更新.env.example
- [ ] 文档化配置选项

### 第四周：生产就绪

- [ ] 添加Dockerfile
- [ ] 添加docker-compose.yml
- [ ] 配置CI/CD
- [ ] 添加pyproject.toml
- [ ] 健康检查和监控
- [ ] 部署文档

## 🎯 预期收益

### 可维护性 ⬆️⬆️⬆️
- 清晰的项目结构
- 统一的代码组织
- 规范的配置管理

### 开发效率 ⬆️⬆️
- 快速定位代码
- 减少重复工作
- 自动化测试和部署

### 团队协作 ⬆️⬆️
- 新人易上手
- 代码规范统一
- 文档完善

### 生产就绪 ⬆️⬆️⬆️
- Docker部署
- CI/CD自动化
- 监控和告警

## 💡 实施建议

### 优先级

**P0 (立即执行):**
1. 根目录清理
2. 测试文件移动
3. 文档整理

**P1 (本周内):**
1. 模块重组决策
2. 配置统一
3. 更新.gitignore

**P2 (两周内):**
1. Docker配置
2. CI/CD配置
3. 生产环境配置

**P3 (一个月内):**
1. 完整文档
2. 监控配置
3. 性能优化

### 风险控制

1. **备份当前代码**
   ```bash
   git tag -a v2.0.0-pre-refactor -m "Before structure refactoring"
   git push origin v2.0.0-pre-refactor
   ```

2. **分支操作**
   ```bash
   git checkout -b refactor/project-structure
   # 在分支上执行重构
   # 完成后合并到main
   ```

3. **增量重构**
   - 不要一次性改太多
   - 每个阶段都确保测试通过
   - 逐步迁移，保持系统可用

## 📝 总结

这个优化方案分为5个阶段：

1. **根目录清理** - 立即可执行，低风险
2. **模块重组** - 需要仔细规划，中风险
3. **配置统一** - 提升可维护性，低风险
4. **生产配置** - 为部署做准备，低风险
5. **文档完善** - 持续改进，低风险

**建议优先执行阶段1和3**，它们风险低、收益高，可以立即改善项目结构。

---

**最后更新**: 2025-10-28
**版本**: 1.0
**状态**: 待执行

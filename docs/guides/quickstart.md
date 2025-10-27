# Academic RAG System - 快速开始指南 🚀

> **5分钟启动你的学术论文问答系统**

---

## 📋 前置检查清单

在开始之前，确认以下条件：

```bash
# ✅ 1. Python 版本检查 (需要 3.9+)
python3 --version
# 应显示: Python 3.13.5 ✓

# ✅ 2. 检查数据文件
ls -lh data/main_system_papers.json
# 应显示: 7.9M 的 JSON 文件 ✓

# ✅ 3. 检查向量数据库
ls vector_db/
# 应显示: chroma.sqlite3 和一些目录 ✓
```

---

## 🎯 方法1: 快速启动（推荐）

### Step 1: 激活虚拟环境

```bash
# 如果已有虚拟环境
source venv_m3max/bin/activate

# 如果没有，创建一个
python3 -m venv venv_m3max
source venv_m3max/bin/activate
pip install -r requirements.txt
```

### Step 2: 启动 Ollama（如果还没运行）

**在另一个终端窗口运行：**
```bash
# 启动 Ollama 服务
ollama serve

# 或者作为后台服务（macOS）
brew services start ollama
```

**下载所需模型：**
```bash
# 下载主要模型（约 4.7GB）
ollama pull llama3.1:8b

# 可选：下载备用模型
ollama pull llama3:8b
ollama pull mistral:7b
```

**验证 Ollama 是否运行：**
```bash
curl http://localhost:11434/api/version
# 应返回: {"version":"0.x.x"}
```

### Step 3: 运行 RAG 系统 ✨

```bash
# 🎯 启动交互式问答系统
python3 main_rag_system.py
```

**预期输出：**
```
=== Academic RAG System Startup ===
Loading papers...
✓ Loaded 59 papers
Initializing vector store...
✓ Vector store ready (1247 chunks)
Loading embedding model...
✓ Model ready: all-mpnet-base-v2
Connecting to LLM...
✓ LLM ready: llama3.1:8b

=== System Ready ===
Commands:
  query: Ask a question
  status: Show system status
  help: Show help
  quit: Exit

>
```

### Step 4: 开始提问 🤔

```
> query: What is the transformer architecture?

🔍 Retrieving relevant documents...
  Found 5 documents (top similarity: 0.89)

🤖 Generating answer...

Answer:
The Transformer is a neural network architecture introduced in the paper
"Attention Is All You Need" (Vaswani et al., 2017). It relies entirely on
self-attention mechanisms to compute representations of input and output
sequences without using recurrence or convolution...

Sources:
  [1] Attention Is All You Need (Vaswani et al., 2017) - Score: 0.89
  [2] BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al.) - Score: 0.85
  ...

> query: How does multi-head attention work?
...

> quit
Goodbye!
```

---

## 🛠️ 方法2: 使用新的 Makefile（推荐给开发者）

```bash
# 查看所有可用命令
make help

# 运行系统（自动激活虚拟环境）
make run

# 或使用新的 CLI 方式（如果已实施 Phase 2）
make query Q="What is BERT?"
```

---

## 📊 常用操作流程

### 完整工作流程

```bash
# 1️⃣ 下载论文（可选，如果需要更新论文库）
python3 collect_classic_papers.py
python3 download_pdfs.py

# 2️⃣ 处理 PDF 并构建索引
python3 process_pdf_fulltext.py

# 3️⃣ 运行问答系统
python3 main_rag_system.py

# 4️⃣ 评估系统性能（可选）
python3 evaluate_rag_system.py
```

### 只需要问答（数据已准备好）

```bash
# 一步到位
source venv_m3max/bin/activate
python3 main_rag_system.py
```

---

## 🔧 故障排查

### 问题 1: `ModuleNotFoundError`

**症状：**
```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**解决：**
```bash
# 激活虚拟环境
source venv_m3max/bin/activate

# 重新安装依赖
pip install -r requirements.txt

# 如果还有问题，手动安装
pip install sentence-transformers chromadb torch ollama
```

### 问题 2: Ollama 连接失败

**症状：**
```
ConnectionError: Could not connect to Ollama at http://localhost:11434
```

**解决：**
```bash
# 检查 Ollama 是否运行
curl http://localhost:11434/api/version

# 如果未运行，启动它
ollama serve

# 或在后台启动
nohup ollama serve > /dev/null 2>&1 &

# 检查模型是否下载
ollama list
```

### 问题 3: 向量数据库为空

**症状：**
```
ValueError: Vector store is empty
```

**解决：**
```bash
# 重建向量数据库
python3 process_pdf_fulltext.py

# 这会：
# 1. 读取 data/raw_papers/*.pdf
# 2. 提取文本和结构
# 3. 智能分块
# 4. 生成嵌入
# 5. 存储到 vector_db/

# 预计时间：10-30分钟（取决于论文数量）
```

### 问题 4: 内存不足

**症状：**
```
RuntimeError: CUDA out of memory
```

**解决：**
```bash
# 在 .env 中设置使用 CPU（如果还没有 .env）
echo "EMBEDDING_DEVICE=cpu" > .env

# 或者减小批处理大小
echo "EMBEDDING_BATCH_SIZE=8" >> .env

# 重启系统
python3 main_rag_system.py
```

### 问题 5: 检索结果不相关

**症状：** 返回的文档与问题不匹配

**解决：**
```bash
# 方法1: 调整检索参数（编辑 main_rag_system.py）
# 找到这些参数并调整：
top_k = 5          # 增加到 10
threshold = 0.5    # 降低到 0.3

# 方法2: 使用更好的查询表述
# 不好: "transformer"
# 更好: "What is the transformer architecture and how does it work?"
```

---

## 📁 项目文件结构说明

```
academic-rag-system/
│
├── main_rag_system.py          ⭐ 主程序入口（问答系统）
├── process_pdf_fulltext.py     📄 PDF 处理和索引构建
├── evaluate_rag_system.py      📊 系统评估
│
├── data/
│   ├── main_system_papers.json     💾 处理后的论文数据（7.9MB）
│   ├── raw_papers/*.pdf            📚 原始 PDF 文件
│   └── embedding_cache/            🗄️ 嵌入缓存
│
├── vector_db/                   🔍 ChromaDB 向量数据库
│   └── chroma.sqlite3
│
├── src/                         📦 核心代码
│   ├── processor/               → PDF 处理、分块
│   ├── retriever/               → 向量检索、混合搜索
│   ├── generator/               → LLM 生成、提示工程
│   ├── embedding/               → 嵌入模型
│   └── evaluation/              → 评估指标
│
├── requirements.txt             📋 依赖列表
├── venv_m3max/                  🐍 虚拟环境
│
└── configs/                     ⚙️ 新增配置系统（v2.0）
    ├── config.yaml
    └── config_loader.py
```

---

## 🎮 交互式命令

系统启动后可以使用以下命令：

```bash
> query: <你的问题>           # 提问
> status                        # 查看系统状态
> help                          # 显示帮助
> quit                          # 退出系统
```

**示例问题：**
```
> query: What are the key innovations in the transformer architecture?
> query: Explain BERT pre-training methodology
> query: Compare ResNet and Vision Transformer
> query: What is attention mechanism in deep learning?
> query: How does GPT differ from BERT?
```

---

## 🚀 高级用法

### 1. 批量查询

创建文件 `queries.txt`:
```
What is the transformer architecture?
How does BERT work?
Explain attention mechanism
```

运行批量查询（需要编写简单脚本）:
```python
# batch_query.py
import sys
sys.path.append('.')
from main_rag_system import MainRAGSystem

system = MainRAGSystem()
system.initialize()

with open('queries.txt') as f:
    for query in f:
        print(f"\n{'='*60}")
        print(f"Q: {query.strip()}")
        print(f"{'='*60}")
        result = system.query(query.strip())
        print(f"A: {result['answer']}\n")
```

### 2. 配置自定义参数

如果使用新配置系统（v2.0）:

```bash
# 复制配置模板
cp .env.example .env

# 编辑配置
vi .env

# 修改这些参数：
RETRIEVAL_TOP_K=10              # 检索文档数量
GENERATION_TEMPERATURE=0.1      # LLM 温度（0=确定性，1=创造性）
EMBEDDING_DEVICE=cpu            # 使用 CPU 或 cuda/mps
OLLAMA_MODEL=llama3.1:8b        # LLM 模型
```

### 3. 性能优化

```bash
# 启用嵌入缓存（加快重复查询）
export ENABLE_EMBEDDING_CACHE=true

# 使用 GPU 加速（如果有）
export EMBEDDING_DEVICE=cuda  # NVIDIA GPU
export EMBEDDING_DEVICE=mps   # Apple Silicon (M1/M2/M3)

# 增加批处理大小（更快但需要更多内存）
export EMBEDDING_BATCH_SIZE=64
```

---

## 📊 系统性能参考

### 典型查询延迟

| 阶段 | 时间 | 说明 |
|------|------|------|
| 检索（BM25 + Vector） | 100-300ms | 从1247个chunks中检索 |
| 重排序（可选） | 50-100ms | Cross-encoder reranking |
| LLM 生成 | 1-3秒 | 取决于答案长度和模型 |
| **总计** | **1.5-3.5秒** | 端到端响应时间 |

### 资源占用

| 资源 | 占用 | 说明 |
|------|------|------|
| 内存（Embedding） | ~2GB | sentence-transformers |
| 内存（VectorDB） | ~500MB | ChromaDB |
| 内存（LLM） | ~5-8GB | Ollama + llama3.1:8b |
| 磁盘（Models） | ~5GB | 模型文件 |
| 磁盘（VectorDB） | ~100MB | 向量数据库 |

---

## 🎯 使用技巧

### 提问技巧

**✅ 好的问题：**
- "Explain the transformer architecture in detail"
- "What are the key differences between BERT and GPT?"
- "How does multi-head attention work in transformers?"

**❌ 避免的问题：**
- "transformer"（太简短）
- "tell me everything"（太宽泛）
- "是什么"（系统主要支持英文，中文效果可能较差）

### 获取更好的答案

1. **明确问题范围**
   ```
   不好: "What is attention?"
   更好: "What is self-attention mechanism in transformer models?"
   ```

2. **包含上下文**
   ```
   不好: "How does it work?"
   更好: "How does BERT pre-training work?"
   ```

3. **要求具体信息**
   ```
   不好: "Tell me about ResNet"
   更好: "What are the key innovations in ResNet compared to previous CNNs?"
   ```

---

## 📈 下一步

### 了解更多

1. **系统架构**: 阅读 `REFACTORING_PLAN.md`
2. **迁移到v2.0**: 阅读 `MIGRATION.md`
3. **自定义配置**: 查看 `configs/config.yaml`
4. **开发指南**: 运行 `make help`

### 扩展功能

```bash
# 添加新论文
cp new_paper.pdf data/raw_papers/
python3 process_pdf_fulltext.py

# 评估系统性能
python3 evaluate_rag_system.py

# 使用新的 Makefile 工具
make test          # 运行测试
make lint          # 代码检查
make format        # 代码格式化
```

---

## 🆘 需要帮助？

### 快速诊断

```bash
# 运行健康检查（如果已设置新系统）
make smoke

# 或手动检查
bash scripts/run_smoke.sh

# 查看系统状态
make status
```

### 常见问题

1. **系统太慢？**
   - 使用 GPU: `export EMBEDDING_DEVICE=cuda`
   - 减少检索数量: 在代码中设置 `top_k=3`

2. **答案质量差？**
   - 尝试不同的模型: `ollama pull mistral:7b`
   - 调整温度参数
   - 使用更具体的问题

3. **找不到相关文档？**
   - 重建索引: `python3 process_pdf_fulltext.py`
   - 检查论文是否在 `data/raw_papers/`

---

## ✅ 快速检查清单

启动系统前确认：

- [ ] Python 3.9+ 已安装
- [ ] 虚拟环境已激活 (`source venv_m3max/bin/activate`)
- [ ] 依赖已安装 (`pip install -r requirements.txt`)
- [ ] Ollama 正在运行 (`curl http://localhost:11434/api/version`)
- [ ] 模型已下载 (`ollama list` 显示 llama3.1:8b)
- [ ] 数据文件存在 (`ls data/main_system_papers.json`)
- [ ] 向量数据库存在 (`ls vector_db/chroma.sqlite3`)

如果全部 ✓，运行：
```bash
python3 main_rag_system.py
```

---

**祝你使用愉快！** 🎉

有问题随时查看：
- 本指南（QUICKSTART.md）
- 详细方案（REFACTORING_PLAN.md）
- 迁移指南（MIGRATION.md）

# 🤖 AI论文RAG系统

一个基于检索增强生成(RAG)技术的AI论文研究助手，专门为研究者提供智能的论文查询和分析服务。

## 🌟 项目特点

- 🔍 **智能检索**: 基于语义向量的精确论文检索
- 🧠 **本地LLM**: 使用Ollama实现完全离线的问答体验
- ⚡ **M3优化**: 专门针对Apple Silicon M3芯片优化
- 📚 **arXiv集成**: 自动从arXiv获取最新AI论文
- 🔧 **模块化**: 清晰的代码结构，易于扩展和维护

## 🚀 快速开始

### 环境要求

- **硬件**: Apple Silicon Mac (推荐M1/M2/M3)
- **系统**: macOS 12.0+
- **Python**: 3.8+
- **内存**: 8GB+ (推荐16GB+)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/你的用户名/ai-paper-rag.git
cd ai-paper-rag
```

2. **创建虚拟环境**
```bash
python3 -m venv venv_m3max
source venv_m3max/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **安装本地LLM**
```bash
# 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载语言模型
ollama pull llama3.1:8b
```

5. **运行系统**
```bash
# 首次运行 - 收集论文和测试系统
python simple_demo.py

# 启动RAG系统
python rag_system.py
```

## 📋 功能演示

### 基础问答
```
🔍 请输入你的问题: 什么是Transformer架构的核心创新？

💡 回答:
基于检索到的论文内容：

📄 论文 1: AUBER: Automated BERT Regularization
👥 作者: Hyun Dong Lee, Seongmin Lee, U Kang
📅 发表: 2020-09-30
🎯 相关度: 0.958
📝 内容摘要: How can we effectively regularize BERT? Although BERT proves its effectiveness in various downstream natural language processing tasks, it often overfits when there are only a small number of training instances...
```

### 支持的查询类型
- 🔬 **技术概念解释**: "什么是注意力机制？"
- 📊 **方法对比**: "CNN和Transformer在视觉任务中的差异"
- 📈 **发展趋势**: "大语言模型的最新进展"
- 🛠️ **实现细节**: "如何实现BERT的预训练？"

## 🏗️ 项目架构

```
ai_paper_rag/
├── data/                    # 数据存储
│   ├── raw_papers/         # 原始PDF文件
│   ├── processed/          # 处理后的数据
│   └── papers_info.json   # 论文元数据
├── src/                    # 源代码
│   ├── collector/          # 论文收集模块
│   ├── processor/          # 文档处理模块
│   ├── retriever/          # 检索模块
│   └── generator/          # 生成模块
├── config/                 # 配置文件
├── vector_db/             # 向量数据库
├── logs/                  # 日志文件
├── requirements.txt       # Python依赖
├── simple_demo.py        # 演示程序
├── rag_system.py         # 主系统
└── README.md             # 项目说明
```

## 🔧 核心技术

### 向量检索
- **嵌入模型**: `all-MiniLM-L6-v2` (轻量级) / `all-mpnet-base-v2` (高性能)
- **向量数据库**: ChromaDB (持久化存储)
- **相似度计算**: 余弦相似度
- **硬件加速**: Apple MPS (Metal Performance Shaders)

### 文档处理
- **PDF解析**: PyMuPDF提取文本内容
- **智能分块**: 基于段落的语义分割
- **元数据提取**: 标题、作者、摘要、发表时间
- **质量过滤**: 去除低质量和重复内容

### 生成模型
- **本地LLM**: Ollama + Llama 3.1
- **无API依赖**: 完全离线运行
- **上下文管理**: 动态上下文长度控制
- **专业提示**: AI研究领域的专门提示词

## 📊 性能指标

### 当前性能 (M3 Max测试)
- **向量化速度**: ~100-200 docs/sec
- **检索延迟**: <100ms
- **端到端响应**: 2-5秒
- **内存使用**: ~2-4GB
- **支持论文数**: 500+ (可扩展到10K+)

### 硬件利用率
- **MPS加速**: ✅ 支持
- **内存优化**: ✅ 大批量处理
- **并发处理**: ✅ 异步I/O

## 🛠️ 配置说明

### 基础配置 (`config/config.yaml`)
```yaml
# 模型配置
models:
  embedding_model: "all-MiniLM-L6-v2"
  local_llm:
    provider: "ollama"
    model: "llama3.1:8b"

# 数据源配置
data_sources:
  arxiv:
    categories: ["cs.AI", "cs.LG", "cs.CV", "cs.CL"]
    max_papers: 100

# 检索配置
retrieval:
  top_k: 5
  chunk_size: 1000
```

### 环境变量 (`.env`)
```bash
# 可选：如果要使用OpenAI API
OPENAI_API_KEY=your_key_here

# 可选：Hugging Face Token
HUGGINGFACE_TOKEN=your_token_here
```

## 🧪 测试和验证

### 运行测试
```bash
# 系统功能测试
python simple_demo.py

# 性能基准测试
python -c "
import time
from src.retriever.vector_store import VectorStore

vs = VectorStore()
start = time.time()
results = vs.search('transformer attention mechanism', top_k=5)
print(f'检索耗时: {(time.time() - start)*1000:.1f}ms')
"
```

### 验证检查清单
- [ ] 论文数据成功收集
- [ ] 向量嵌入正常生成
- [ ] 本地LLM连接正常
- [ ] 检索结果相关性高
- [ ] 回答质量满足预期

## 📈 开发路线图

### ✅ 已完成
- [x] 基础RAG架构
- [x] arXiv论文收集
- [x] 向量检索系统
- [x] 本地LLM集成
- [x] 简单问答界面

### 🚧 开发中
- [ ] 智能分块优化
- [ ] 混合检索算法
- [ ] 重排序模型
- [ ] 查询理解增强

### 📋 计划中
- [ ] Advanced RAG技术 (Self-RAG, RAG-Fusion)
- [ ] 知识图谱集成
- [ ] 多模态支持 (图表、公式)
- [ ] Web界面开发
- [ ] 性能监控系统

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范
- 使用 Black 格式化代码
- 添加类型注解
- 编写单元测试
- 更新文档

## 📝 更新日志

### v0.1.0 (2024-01-XX)
- ✨ 初始版本发布
- 🔍 基础向量检索功能
- 🤖 本地LLM集成
- 📚 arXiv论文自动收集
- ⚡ Apple M3优化

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - 提供优秀的嵌入模型
- [ChromaDB](https://github.com/chroma-core/chroma) - 高性能向量数据库
- [Ollama](https://github.com/ollama/ollama) - 本地LLM运行环境
- [arXiv](https://arxiv.org/) - 开放的学术论文资源

## 📞 联系方式

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/你的用户名/ai-paper-rag/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/你的用户名/ai-paper-rag/discussions)

---

⭐ **如果这个项目对你有帮助，请给它一个星标！**

[![GitHub stars](https://img.shields.io/github/stars/HaojingTong/ai-paper-rag.svg?style=social&label=Star)](https://github.com/HaojingTong/ai-paper-rag)
[![GitHub forks](https://img.shields.io/github/forks/HaojingTong/ai-paper-rag.svg?style=social&label=Fork)](https://github.com/HaojingTong/ai-paper-rag/fork)

# 🤖 学术论文RAG系统 (Academic RAG System)

一个基于检索增强生成(RAG)技术的智能学术研究助手，专门为AI研究者提供精准的论文查询、分析和问答服务。

## 🌟 核心特性

- 🔍 **智能检索**: 基于语义向量的精确论文检索，支持多策略混合检索
- 🧠 **本地LLM**: 使用Ollama + Llama 3.1实现完全离线的问答体验
- 📄 **全文处理**: 完整PDF内容提取，智能章节识别和语义分块
- 🎯 **上下文增强**: 智能提示工程，多模式查询理解和上下文优化
- 📊 **科学评估**: 完整的RAG系统评估框架，多维度性能量化
- ⚡ **高性能**: 针对Apple Silicon优化，支持MPS硬件加速
- 📚 **数据丰富**: 集成59篇高质量AI论文，支持扩展到更大规模

## 🚀 快速开始

### 环境要求

- **硬件**: Apple Silicon Mac (推荐M1/M2/M3) 或 x86_64
- **系统**: macOS 12.0+ / Linux / Windows
- **Python**: 3.8+
- **内存**: 8GB+ (推荐16GB+)
- **存储**: 5GB+ 空闲空间

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/HaojingTong/academic-rag-system.git
cd academic-rag-system
```

2. **创建虚拟环境**
```bash
python3 -m venv venv_m3max
source venv_m3max/bin/activate  # macOS/Linux
# venv_m3max\Scripts\activate  # Windows
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

# 启动Ollama服务
ollama serve
```

5. **处理论文数据**
```bash
# 处理PDF全文内容 (首次运行必须)
python process_pdf_fulltext.py
```

6. **启动RAG系统**
```bash
# 运行主RAG系统
python main_rag_system.py
```

## 📋 系统演示

### 智能问答示例

```
🤖 学术论文RAG助手 - 启动中...
================================================================================
✅ 初始化完成! 

🔍 请输入你的问题 (输入 'quit' 退出): 什么是Transformer架构的核心创新？

🚀 开始智能检索...
🔍 检索策略: 混合检索 (向量+BM25)
📊 找到相关文档: 8 个

💡 生成回答中...

📖 回答:
Transformer架构的核心创新主要体现在以下几个方面:

1. **自注意力机制 (Self-Attention)**: 完全摒弃了循环和卷积结构，仅使用注意力机制来建模序列内部的依赖关系。

2. **并行计算**: 与RNN不同，Transformer可以并行处理序列中的所有位置，大大提高了训练效率。

3. **多头注意力**: 通过多个注意力头来捕捉不同类型的依赖关系，增强模型的表达能力。

4. **位置编码**: 通过位置编码来保留序列的位置信息，解决了注意力机制无法感知位置的问题。

📚 参考来源:
   📄 Attention Is All You Need (相关度: 0.94)
   📄 BERT: Pre-training of Deep Bidirectional Transformers (相关度: 0.89)
   📄 Language Models are Few-Shot Learners (相关度: 0.85)

⏱️ 处理时间: 3.2秒
```

### 支持的查询类型

- 🔬 **技术概念解释**: "什么是注意力机制？"、"解释BERT的预训练过程"
- 📊 **方法对比**: "Transformer和RNN的区别"、"GAN和VAE的优缺点"  
- 📈 **发展趋势**: "大语言模型的最新进展"、"计算机视觉的发展历程"
- 🛠️ **实现细节**: "如何训练GPT模型？"、"ResNet的残差连接原理"
- 🧐 **深度分析**: "为什么Transformer能够取代RNN？"、"自监督学习的优势"

## 🏗️ 系统架构

```
academic-rag-system/
├── data/                           # 数据存储
│   ├── raw_papers/                 # 原始PDF文件 (59篇高质量论文)
│   ├── embedding_cache/            # 嵌入向量缓存
│   ├── evaluation/                 # 评估数据和结果
│   ├── main_system_papers.json     # 完整论文数据 (包含全文分块)
│   └── high_quality_papers.json    # 高质量论文清单
├── src/                            # 核心源代码
│   ├── config/                     # 配置管理
│   │   └── embedding_config.py     # 嵌入模型配置
│   ├── embedding/                  # 文本嵌入模块  
│   │   └── advanced_embedding.py   # 高级嵌入处理
│   ├── generator/                  # 文本生成模块
│   │   ├── llm_client.py          # LLM客户端
│   │   ├── prompt_engineering.py   # 智能提示工程
│   │   └── quality_enhancement.py  # 回答质量增强
│   ├── processor/                  # 文档处理模块
│   │   ├── pdf_processor.py       # 学术PDF完整处理
│   │   └── document_chunker.py     # 智能文档分块
│   ├── retriever/                  # 检索模块
│   │   ├── advanced_retrieval.py   # 高级检索策略
│   │   ├── enhanced_vector_retrieval.py # 增强向量检索
│   │   └── enhanced_vector_store.py     # 向量存储管理
│   └── evaluation/                 # 评估框架 ⭐ NEW
│       ├── evaluation_metrics.py   # 核心评估指标
│       ├── benchmark_datasets.py   # 基准数据集管理  
│       └── rag_evaluator.py       # 完整评估器
├── vector_db/                      # 向量数据库 (ChromaDB)
├── main_rag_system.py             # 🎯 主系统入口
├── process_pdf_fulltext.py        # 📄 PDF全文处理脚本
├── evaluate_rag_system.py         # 📊 系统评估脚本
├── collect_classic_papers.py      # 📚 经典论文收集
├── requirements.txt               # Python依赖
├── EVALUATION_FRAMEWORK.md       # 📋 评估框架文档
└── README.md                      # 项目说明
```

## 🔧 核心技术栈

### 🧠 智能检索层
- **嵌入模型**: `all-MiniLM-L6-v2` (轻量高效) / `all-mpnet-base-v2` (高性能)
- **向量数据库**: ChromaDB (持久化 + 高性能)
- **检索策略**: 混合检索 (语义向量 + BM25关键词)
- **重排序**: 基于相关性的智能重排序算法

### 📄 文档处理层
- **PDF解析**: PyMuPDF + 智能内容清理
- **章节识别**: 基于模式匹配的学术论文结构识别
- **智能分块**: 语义感知的文档切分 (600字符块，100字符重叠)
- **元数据增强**: 标题、作者、摘要、章节类型等丰富元数据

### 🎯 生成增强层  
- **本地LLM**: Ollama + Llama 3.1:8b (完全离线)
- **提示工程**: 智能查询分类和上下文构建
- **质量增强**: 多层次回答优化和验证
- **上下文管理**: 动态上下文长度控制 (最大4000字符)

### 📊 评估分析层 ⭐ NEW
- **多维评估**: 上下文相关性、答案忠实度、答案相关性等5大维度
- **自动数据集**: 基于论文内容自动生成评估问答对
- **可视化报告**: 自动生成性能图表和分析报告
- **持续优化**: 基于评估结果的系统改进建议

## 📊 性能指标

### 系统性能 (M3 Max测试)
| 指标 | 性能 | 说明 |
|------|------|------|
| 🔍 检索延迟 | <200ms | 单次语义检索平均耗时 |
| 🧠 生成延迟 | 2-5秒 | 端到端问答响应时间 |
| 💾 内存使用 | 3-6GB | 包含模型和向量数据库 |
| 📚 支持论文 | 59篇+ | 当前高质量论文数量 |
| 📄 处理能力 | 6.8万+ | 总处理文档块数量 |

### RAG系统评估结果
| 评估维度 | 评分 | 等级 |
|----------|------|------|
| 🎯 综合评分 | 0.72-0.85 | 良好-优秀 |
| 🔍 上下文相关性 | 0.68-0.82 | 检索质量高 |
| 🤝 答案忠实度 | 0.75-0.88 | 基于上下文生成 |
| 💡 答案相关性 | 0.70-0.85 | 直接回答问题 |
| 📈 上下文精确度 | 0.65-0.80 | 检索精确度良好 |

## 🛠️ 高级功能

### 1. 📊 系统评估

```bash
# 快速评估 (使用示例数据集)
python evaluate_rag_system.py --quick

# 完整评估 (生成真实评估数据)
python evaluate_rag_system.py

# 查看评估报告
ls data/evaluation/results/
```

评估维度包括:
- **上下文相关性**: 检索内容与问题的相关程度
- **答案忠实度**: 答案基于上下文的程度  
- **答案相关性**: 答案回答问题的程度
- **上下文精确度**: 检索内容的准确性
- **上下文召回率**: 相关文档的覆盖率

### 2. 🎯 智能提示工程

系统自动识别查询类型并优化提示:
- **一般查询**: 基础信息检索
- **技术查询**: 深入技术细节
- **对比查询**: 多概念对比分析  
- **定义查询**: 概念定义和解释
- **应用查询**: 实际应用场景

### 3. 🔄 混合检索策略

```python
# 自动选择最佳检索策略
retrieval_strategies = [
    "vector_only",      # 纯向量检索
    "bm25_only",        # 纯关键词检索  
    "hybrid",           # 混合检索 (默认)
    "multi_query"       # 多查询扩展
]
```

### 4. 📈 质量增强

- **上下文优化**: 智能上下文选择和排序
- **答案验证**: 多层次答案质量检查
- **引用管理**: 自动生成参考文献
- **格式美化**: 结构化答案呈现

## 🧪 使用指南

### 基础使用

```bash
# 启动交互式问答
python main_rag_system.py

# 处理新的PDF论文
python process_pdf_fulltext.py

# 收集更多论文  
python collect_classic_papers.py
```

### 编程接口

```python
from main_rag_system import AcademicRAGSystem

# 初始化系统
rag = AcademicRAGSystem()

# 查询论文
response = rag.query_academic_papers("什么是注意力机制？")

# 获取详细结果
print(f"答案: {response['answer']}")
print(f"来源: {len(response['final_results'])} 篇相关论文")
```

### 评估接口

```python
from src.evaluation import RAGEvaluator, BenchmarkDatasets

# 创建评估器
evaluator = RAGEvaluator()

# 创建测试数据集
dataset_manager = BenchmarkDatasets()
test_dataset = dataset_manager.create_sample_dataset()

# 评估RAG系统
results = evaluator.evaluate_rag_system(rag, test_dataset)
```

## 📈 开发路线图

### ✅ 已完成 (v1.0)
- [x] 核心RAG架构
- [x] PDF全文处理系统
- [x] 智能提示工程
- [x] 混合检索策略  
- [x] 完整评估框架
- [x] 59篇高质量论文集成
- [x] 本地LLM集成

### 🚧 开发中 (v1.1)
- [ ] Web界面开发
- [ ] 实时论文更新
- [ ] 多语言支持
- [ ] 知识图谱构建

### 📋 计划中 (v2.0)
- [ ] Advanced RAG技术 (Self-RAG, RAG-Fusion)
- [ ] 多模态支持 (图表、公式理解)
- [ ] 协作功能 (多用户、共享)
- [ ] 移动端应用

## 📊 数据集

### 当前论文集合 (59篇精选)
包含AI领域的经典和前沿论文:

**深度学习基础**:
- BERT, GPT, Transformer系列
- ResNet, DenseNet等视觉模型
- Adam, Dropout等训练技术

**前沿进展**:  
- GPT-3, PaLM等大语言模型
- Vision Transformer, CLIP等多模态
- Self-supervised Learning最新成果

**评估能力**:
- 📊 平均每篇论文: 118.6个语义块
- 🔍 总检索单元: 6,800+ 文档块
- 📈 覆盖范围: NLP、CV、ML等主要AI分支

## 🔧 配置说明

### 环境配置
```bash
# 可选环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export OLLAMA_HOST="http://localhost:11434"
```

### 模型配置
```python
# 嵌入模型选择
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # 轻量高效
# EMBEDDING_MODEL = "all-mpnet-base-v2"  # 高性能

# LLM模型配置  
LLM_MODEL = "llama3.1:8b"               # 默认模型
# LLM_MODEL = "llama3.1:70b"            # 高精度模型
```

## 🧪 测试和验证

### 功能测试

```bash
# 测试评估框架
python -c "
from src.evaluation.evaluation_metrics import EvaluationMetrics
evaluator = EvaluationMetrics()
print('✅ 评估框架正常')
"

# 测试RAG系统
python -c "
from main_rag_system import AcademicRAGSystem  
rag = AcademicRAGSystem()
print('✅ RAG系统正常')
"
```

### 性能基准

```bash
# 检索性能测试
python -c "
import time
from main_rag_system import AcademicRAGSystem
rag = AcademicRAGSystem()
start = time.time()
response = rag.query_academic_papers('transformer attention')
print(f'响应时间: {time.time()-start:.2f}秒')
"
```

## 🔍 故障排除

### 常见问题

1. **Ollama连接失败**
```bash
# 检查Ollama状态
curl http://localhost:11434/api/tags

# 重启Ollama
ollama serve
```

2. **PDF处理失败**  
```bash
# 检查PDF文件
ls -la data/raw_papers/

# 重新处理PDF
python process_pdf_fulltext.py
```

3. **评估框架错误**
```bash
# 安装缺失依赖
pip install matplotlib seaborn

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### 性能优化

```python
# 减少内存使用
embedding_config = {
    "batch_size": 32,      # 减少批次大小
    "max_length": 512,     # 限制序列长度
    "device": "mps"        # 使用MPS加速 (Mac)
}

# 提高检索速度
retrieval_config = {
    "top_k": 3,           # 减少检索数量
    "use_reranker": False # 禁用重排序
}
```

## 🤝 贡献指南

欢迎参与项目开发！请遵循以下流程:

1. **Fork** 项目到你的GitHub
2. **创建** 特性分支 (`git checkout -b feature/AmazingFeature`)
3. **提交** 更改 (`git commit -m 'Add AmazingFeature'`)
4. **推送** 分支 (`git push origin feature/AmazingFeature`)
5. **创建** Pull Request

### 开发规范
- 使用类型注解 (`typing`)
- 遵循PEP 8代码风格
- 编写单元测试
- 更新相关文档
- 测试性能影响

### 贡献方向
- 🔍 **检索优化**: 改进检索算法和策略
- 📊 **评估增强**: 扩展评估指标和方法
- 🎯 **提示优化**: 改进提示工程技术
- 🌐 **界面开发**: Web/移动端界面
- 📚 **数据扩展**: 更多高质量论文数据

## 📝 更新日志

### v1.0.0 (2024-09-08)
- ✨ 完整RAG系统发布
- 📄 PDF全文处理系统
- 🎯 智能提示工程  
- 📊 完整评估框架
- 🔍 混合检索策略
- 📚 59篇高质量论文集成

### v0.9.0 (2024-09-07)  
- 🧠 本地LLM集成
- 🔧 系统架构重构
- ⚡ Apple Silicon优化

### v0.8.0 (2024-09-06)
- 🔍 基础向量检索
- 📄 PDF处理功能
- 📚 arXiv数据收集

## 📚 参考文献

- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** - Lewis et al.
- **Attention Is All You Need** - Vaswani et al. 
- **BERT: Pre-training of Deep Bidirectional Transformers** - Devlin et al.
- **Language Models are Few-Shot Learners** - Brown et al.
- **RAGAS: Automated Evaluation of Retrieval Augmented Generation** - Es et al.

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持:
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - 优秀的嵌入模型库
- [ChromaDB](https://github.com/chroma-core/chroma) - 高性能向量数据库
- [Ollama](https://github.com/ollama/ollama) - 本地LLM运行环境
- [arXiv](https://arxiv.org/) - 开放学术资源平台
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF处理工具

## 📞 联系方式

- 📧 **Email**: haojing.tong@outlook.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/HaojingTong/academic-rag-system/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/HaojingTong/academic-rag-system/discussions)
- 📝 **Blog**: 技术博客即将推出

---

⭐ **如果这个项目对你的研究有帮助，请给它一个星标！**

[![GitHub stars](https://img.shields.io/github/stars/HaojingTong/academic-rag-system.svg?style=social&label=Star)](https://github.com/HaojingTong/academic-rag-system)
[![GitHub forks](https://img.shields.io/github/forks/HaojingTong/academic-rag-system.svg?style=social&label=Fork)](https://github.com/HaojingTong/academic-rag-system/fork)

💡 **让AI研究更高效，让知识获取更智能！**
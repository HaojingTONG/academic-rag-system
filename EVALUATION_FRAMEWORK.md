# RAG系统评估框架 (Evaluation Framework)

## 📊 概述

本评估框架为学术论文RAG系统提供了完整的性能评估解决方案，从多个维度量化RAG系统的表现，帮助识别改进方向。

## 🎯 核心评估指标

### 1. **上下文相关性 (Context Relevance)**
- **定义**: 检索到的上下文与用户问题的相关程度
- **计算方法**: 使用语义相似度计算问题与上下文的匹配度
- **评分范围**: 0-1（1分为完全相关）

### 2. **答案忠实度 (Faithfulness)**
- **定义**: 生成的答案是否完全基于提供的上下文，没有虚构信息
- **计算方法**: 结合语义相似度和词汇重叠度分析
- **评分范围**: 0-1（1分为完全忠实）

### 3. **答案相关性 (Answer Relevance)**
- **定义**: 答案是否直接回答了用户的问题
- **计算方法**: 计算问题与答案的语义相似度
- **评分范围**: 0-1（1分为完全相关）

### 4. **上下文精确度 (Context Precision)**
- **定义**: 检索到的上下文中有多少是真正相关的
- **计算方法**: 相关上下文数量 / 总检索上下文数量
- **评分范围**: 0-1（1分为完全精确）

### 5. **上下文召回率 (Context Recall)**
- **定义**: 相关文档中有多少被成功检索到
- **计算方法**: 检索到的相关文档 / 总相关文档（需要标准答案）
- **评分范围**: 0-1（1分为完全召回）

## 🏗️ 框架架构

```
src/evaluation/
├── __init__.py                 # 模块初始化
├── evaluation_metrics.py      # 核心评估指标实现
├── benchmark_datasets.py      # 基准数据集管理
└── rag_evaluator.py           # 完整评估器
```

## 🚀 快速开始

### 1. 基础评估

```python
from src.evaluation import RAGEvaluator, BenchmarkDatasets

# 初始化评估器
evaluator = RAGEvaluator()

# 创建测试数据集
dataset_manager = BenchmarkDatasets()
test_dataset = dataset_manager.create_sample_dataset()

# 评估RAG系统
results = evaluator.evaluate_rag_system(
    rag_system=your_rag_system,
    evaluation_dataset=test_dataset
)
```

### 2. 使用评估脚本

```bash
# 完整评估
python evaluate_rag_system.py

# 快速评估
python evaluate_rag_system.py --quick
```

## 📋 评估数据集

### 数据集类型

1. **示例数据集**: 预定义的测试案例，适合快速测试
2. **生成数据集**: 基于论文内容自动生成问答对
3. **自定义数据集**: 手动创建的专门测试案例

### 数据集结构

```python
@dataclass
class EvaluationCase:
    id: str                     # 案例ID
    question: str               # 问题
    contexts: List[str]         # 检索到的上下文
    ground_truth_answer: str    # 标准答案
    relevant_contexts: List[str] # 相关上下文（用于召回率）
    difficulty: str             # 难度等级: easy, medium, hard
    category: str               # 问题类别
    metadata: Dict              # 额外元数据
```

## 📊 评估报告

评估完成后会生成多种格式的报告：

### 1. JSON报告
- **详细结果**: `evaluation_detailed_YYYYMMDD_HHMMSS.json`
- **摘要报告**: `evaluation_summary_YYYYMMDD_HHMMSS.json`

### 2. CSV报告
- **表格数据**: `evaluation_report_YYYYMMDD_HHMMSS.csv`
- 便于Excel分析和处理

### 3. 可视化图表
- **性能图表**: `evaluation_charts_YYYYMMDD_HHMMSS.png`
- 包含分数分布、指标对比、相关性分析等

## 🔧 高级功能

### 1. LLM深度评估

启用LLM进行更深入的质量评估：

```python
# 启用LLM评估
results = evaluator.evaluate_rag_system(
    rag_system=rag_system,
    evaluation_dataset=dataset,
    use_llm_evaluation=True  # 启用LLM评估
)
```

LLM评估维度：
- 上下文相关性
- 答案忠实度  
- 答案相关性
- 答案完整性
- 答案清晰度

### 2. 自定义评估指标

```python
class CustomEvaluationMetrics(EvaluationMetrics):
    def custom_metric(self, question: str, answer: str) -> float:
        # 实现自定义评估逻辑
        return custom_score

evaluator = RAGEvaluator(custom_metrics=CustomEvaluationMetrics())
```

### 3. 批量评估

```python
# 评估多个RAG系统配置
configurations = [
    {"embedding_model": "all-MiniLM-L6-v2", "chunk_size": 500},
    {"embedding_model": "all-mpnet-base-v2", "chunk_size": 1000}
]

for config in configurations:
    rag_system = AcademicRAGSystem(**config)
    results = evaluator.evaluate_rag_system(rag_system, dataset)
```

## 📈 评估解读

### 综合评分标准

- **0.8-1.0**: 优秀 🏆 - 系统表现出色
- **0.7-0.8**: 良好 👍 - 系统表现良好
- **0.6-0.7**: 中等 👌 - 系统表现一般
- **0.5-0.6**: 及格 📝 - 系统基本可用
- **0.0-0.5**: 需要改进 📉 - 系统需要优化

### 指标分析指南

1. **上下文相关性低** → 优化检索算法、改进查询理解
2. **答案忠实度低** → 改进提示工程、增强上下文利用
3. **答案相关性低** → 优化问答匹配、改进生成策略
4. **上下文精确度低** → 提高检索精确度、过滤无关内容
5. **上下文召回率低** → 扩展检索范围、改进索引策略

## 🔄 持续改进流程

1. **基线评估**: 建立当前系统性能基线
2. **问题识别**: 分析薄弱环节和改进机会
3. **系统优化**: 针对性改进RAG系统组件
4. **效果验证**: 重新评估验证改进效果
5. **迭代优化**: 重复上述流程持续改进

## 🛠️ API参考

### EvaluationMetrics类

```python
class EvaluationMetrics:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2")
    
    def evaluate_single_case(self, 
                           question: str,
                           retrieved_contexts: List[str],
                           generated_answer: str,
                           ground_truth: Optional[str] = None,
                           relevant_contexts: Optional[List[str]] = None
                          ) -> EvaluationResult
```

### BenchmarkDatasets类

```python
class BenchmarkDatasets:
    def __init__(self, data_dir: str = "data/evaluation")
    
    def generate_academic_qa_dataset(self, 
                                   papers_data_path: str, 
                                   num_questions: int = 100
                                  ) -> List[EvaluationCase]
    
    def create_custom_dataset(self, cases: List[Dict]) -> List[EvaluationCase]
    
    def save_dataset(self, dataset: List[EvaluationCase], filename: str)
    
    def load_dataset(self, filename: str) -> List[EvaluationCase]
```

### RAGEvaluator类

```python
class RAGEvaluator:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_base_url: str = "http://localhost:11434",
                 llm_model: str = "llama3.1:8b",
                 output_dir: str = "data/evaluation/results")
    
    def evaluate_rag_system(self, 
                           rag_system,
                           evaluation_dataset: List[EvaluationCase],
                           use_llm_evaluation: bool = True,
                           save_results: bool = True
                          ) -> Dict[str, Any]
```

## 📝 最佳实践

1. **定期评估**: 建议定期（如每月）评估系统性能
2. **多样化测试**: 使用不同难度和类型的测试案例
3. **A/B测试**: 对比不同配置的性能差异
4. **用户反馈**: 结合实际用户反馈进行评估
5. **持续监控**: 在生产环境中持续监控关键指标

## 🔍 故障排除

### 常见问题

1. **评估速度慢**: 禁用LLM评估或使用更快的嵌入模型
2. **内存不足**: 减少批次大小或使用更小的数据集  
3. **LLM连接失败**: 检查Ollama服务状态
4. **依赖缺失**: 安装matplotlib和seaborn

### 性能优化

```python
# 使用更快的嵌入模型
evaluator = RAGEvaluator(embedding_model="all-MiniLM-L6-v2")

# 禁用LLM评估以提高速度
results = evaluator.evaluate_rag_system(
    rag_system=rag_system,
    evaluation_dataset=dataset,
    use_llm_evaluation=False
)
```

## 📚 参考资源

- **RAGAs Framework**: https://github.com/explodinggradients/ragas
- **TruLens**: https://github.com/truera/trulens
- **Sentence Transformers**: https://www.sbert.net/
- **相关论文**: 
  - "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
  - "Evaluating Retrieval-Augmented Generation Systems"

---

通过这个评估框架，你可以科学地量化RAG系统的性能，识别改进方向，并跟踪优化效果。建议从示例数据集开始，逐步扩展到更复杂的评估场景。
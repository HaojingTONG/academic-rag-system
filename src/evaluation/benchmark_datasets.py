# src/evaluation/benchmark_datasets.py
"""
评估基准数据集管理
生成和管理用于RAG系统评估的问答数据集
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import requests

@dataclass
class EvaluationCase:
    """评估案例数据结构"""
    id: str
    question: str
    contexts: List[str]          # 检索到的上下文
    ground_truth_answer: str     # 标准答案
    relevant_contexts: List[str] # 相关的上下文（用于召回率计算）
    difficulty: str              # 难度等级: easy, medium, hard
    category: str                # 问题类别
    metadata: Dict               # 额外元数据

class BenchmarkDatasets:
    """基准数据集管理器"""
    
    def __init__(self, data_dir: str = "data/evaluation"):
        """
        初始化数据集管理器
        
        Args:
            data_dir: 评估数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        print(f"✅ 初始化评估数据集管理器: {data_dir}")
    
    def generate_academic_qa_dataset(self, papers_data_path: str, 
                                   num_questions: int = 100,
                                   llm_base_url: str = "http://localhost:11434",
                                   model: str = "llama3.1:8b") -> List[EvaluationCase]:
        """
        基于学术论文生成问答评估数据集
        
        Args:
            papers_data_path: 论文数据文件路径
            num_questions: 生成问题数量
            llm_base_url: LLM服务URL
            model: LLM模型名称
            
        Returns:
            List[EvaluationCase]: 评估案例列表
        """
        print(f"🔄 生成学术问答数据集: {num_questions} 个问题...")
        
        # 加载论文数据
        with open(papers_data_path, 'r', encoding='utf-8') as f:
            papers_data = json.load(f)
        
        evaluation_cases = []
        
        for i in range(num_questions):
            print(f"   生成问题 {i+1}/{num_questions}...")
            
            # 随机选择一篇论文
            paper = random.choice(papers_data)
            
            # 选择论文的一些文档块作为上下文
            chunks = paper.get('processed_chunks', [])
            if not chunks:
                continue
                
            # 随机选择2-4个文档块
            num_chunks = min(random.randint(2, 4), len(chunks))
            selected_chunks = random.sample(chunks, num_chunks)
            
            # 提取上下文文本
            contexts = [chunk['text'] for chunk in selected_chunks]
            paper_title = paper.get('title', 'Unknown Paper')
            
            # 使用LLM生成问题和答案
            qa_pair = self._generate_qa_with_llm(
                paper_title, contexts, llm_base_url, model
            )
            
            if qa_pair:
                # 确定问题难度和类别
                difficulty = self._determine_difficulty(qa_pair['question'])
                category = self._determine_category(qa_pair['question'])
                
                case = EvaluationCase(
                    id=f"academic_qa_{i+1}",
                    question=qa_pair['question'],
                    contexts=contexts,
                    ground_truth_answer=qa_pair['answer'],
                    relevant_contexts=contexts,  # 假设选择的上下文都是相关的
                    difficulty=difficulty,
                    category=category,
                    metadata={
                        'paper_id': paper.get('id', ''),
                        'paper_title': paper_title,
                        'chunk_count': len(contexts)
                    }
                )
                
                evaluation_cases.append(case)
                
            # 避免请求过于频繁
            import time
            time.sleep(1)
        
        print(f"✅ 成功生成 {len(evaluation_cases)} 个评估案例")
        return evaluation_cases
    
    def _generate_qa_with_llm(self, paper_title: str, contexts: List[str], 
                             llm_base_url: str, model: str) -> Optional[Dict]:
        """使用LLM基于论文内容生成问答对"""
        
        combined_context = "\n\n".join(contexts)
        
        prompt = f"""
基于以下学术论文内容，生成一个高质量的问答对。问题应该具有挑战性，答案应该完全基于给定的上下文。

论文标题: {paper_title}

论文内容:
{combined_context}

请生成：
1. 一个具体、有挑战性的问题（避免过于宽泛的问题）
2. 一个详细、准确的答案（完全基于上述内容）

返回格式：
{{
  "question": "<问题>",
  "answer": "<答案>"
}}
"""
        
        try:
            response = requests.post(
                f"{llm_base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer_text = result.get('response', '')
                
                # 尝试解析JSON
                try:
                    json_start = answer_text.find('{')
                    json_end = answer_text.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_str = answer_text[json_start:json_end]
                        qa_pair = json.loads(json_str)
                        
                        # 验证问答对质量
                        if (qa_pair.get('question') and qa_pair.get('answer') and
                            len(qa_pair['question']) > 10 and len(qa_pair['answer']) > 20):
                            return qa_pair
                except:
                    pass
                    
        except Exception as e:
            print(f"      ⚠️ LLM生成失败: {e}")
            
        return None
    
    def _determine_difficulty(self, question: str) -> str:
        """根据问题内容判断难度等级"""
        question_lower = question.lower()
        
        # 简单问题关键词
        easy_indicators = ['what is', 'define', 'list', 'name', 'who', 'when', 'where']
        
        # 困难问题关键词  
        hard_indicators = ['compare', 'analyze', 'evaluate', 'why', 'how does', 'what are the implications', 'discuss']
        
        if any(indicator in question_lower for indicator in hard_indicators):
            return 'hard'
        elif any(indicator in question_lower for indicator in easy_indicators):
            return 'easy'
        else:
            return 'medium'
    
    def _determine_category(self, question: str) -> str:
        """根据问题内容判断类别"""
        question_lower = question.lower()
        
        categories = {
            'definition': ['what is', 'define', 'definition of'],
            'method': ['how', 'method', 'approach', 'algorithm', 'technique'],
            'comparison': ['compare', 'difference', 'vs', 'versus', 'better'],
            'application': ['application', 'use', 'apply', 'implement'],
            'evaluation': ['performance', 'result', 'accuracy', 'evaluation', 'experiment'],
            'conceptual': ['why', 'reason', 'principle', 'concept', 'theory']
        }
        
        for category, keywords in categories.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
                
        return 'general'
    
    def create_custom_dataset(self, cases: List[Dict]) -> List[EvaluationCase]:
        """
        从自定义数据创建评估数据集
        
        Args:
            cases: 自定义案例列表
            
        Returns:
            List[EvaluationCase]: 评估案例列表
        """
        evaluation_cases = []
        
        for i, case in enumerate(cases):
            eval_case = EvaluationCase(
                id=case.get('id', f"custom_{i+1}"),
                question=case['question'],
                contexts=case['contexts'],
                ground_truth_answer=case['ground_truth_answer'],
                relevant_contexts=case.get('relevant_contexts', case['contexts']),
                difficulty=case.get('difficulty', 'medium'),
                category=case.get('category', 'general'),
                metadata=case.get('metadata', {})
            )
            evaluation_cases.append(eval_case)
            
        return evaluation_cases
    
    def save_dataset(self, dataset: List[EvaluationCase], filename: str):
        """保存数据集到文件"""
        dataset_data = []
        for case in dataset:
            case_dict = {
                'id': case.id,
                'question': case.question,
                'contexts': case.contexts,
                'ground_truth_answer': case.ground_truth_answer,
                'relevant_contexts': case.relevant_contexts,
                'difficulty': case.difficulty,
                'category': case.category,
                'metadata': case.metadata
            }
            dataset_data.append(case_dict)
        
        save_path = self.data_dir / filename
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 数据集已保存: {save_path}")
    
    def load_dataset(self, filename: str) -> List[EvaluationCase]:
        """从文件加载数据集"""
        load_path = self.data_dir / filename
        
        if not load_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {load_path}")
        
        with open(load_path, 'r', encoding='utf-8') as f:
            dataset_data = json.load(f)
        
        evaluation_cases = []
        for case_dict in dataset_data:
            case = EvaluationCase(
                id=case_dict['id'],
                question=case_dict['question'],
                contexts=case_dict['contexts'],
                ground_truth_answer=case_dict['ground_truth_answer'],
                relevant_contexts=case_dict['relevant_contexts'],
                difficulty=case_dict['difficulty'],
                category=case_dict['category'],
                metadata=case_dict['metadata']
            )
            evaluation_cases.append(case)
        
        print(f"✅ 数据集已加载: {load_path} ({len(evaluation_cases)} 个案例)")
        return evaluation_cases

    def create_sample_dataset(self) -> List[EvaluationCase]:
        """创建示例评估数据集"""
        sample_cases = [
            {
                'id': 'sample_1',
                'question': 'What is the main contribution of the attention mechanism in neural networks?',
                'contexts': [
                    'The attention mechanism allows neural networks to focus on relevant parts of the input sequence.',
                    'Attention was introduced to solve the bottleneck problem in encoder-decoder architectures.',
                    'The mechanism computes weighted averages of input representations based on relevance scores.'
                ],
                'ground_truth_answer': 'The main contribution of the attention mechanism is allowing neural networks to selectively focus on relevant parts of the input sequence, solving the bottleneck problem in encoder-decoder architectures by computing weighted averages based on relevance scores.',
                'difficulty': 'medium',
                'category': 'conceptual'
            },
            {
                'id': 'sample_2', 
                'question': 'How does the transformer architecture differ from RNN-based models?',
                'contexts': [
                    'Transformers rely entirely on self-attention mechanisms without recurrence.',
                    'RNNs process sequences step by step, leading to sequential dependencies.',
                    'The transformer architecture enables parallel processing of all sequence positions.',
                    'Self-attention allows direct modeling of dependencies regardless of distance.'
                ],
                'ground_truth_answer': 'The transformer architecture differs from RNN-based models by relying entirely on self-attention mechanisms without recurrence, enabling parallel processing of all sequence positions and direct modeling of long-range dependencies, unlike RNNs which process sequences sequentially.',
                'difficulty': 'hard',
                'category': 'comparison'
            }
        ]
        
        return self.create_custom_dataset(sample_cases)

# 使用示例
def test_benchmark_datasets():
    """测试基准数据集功能"""
    print("🧪 测试基准数据集...")
    
    dataset_manager = BenchmarkDatasets()
    
    # 创建示例数据集
    sample_dataset = dataset_manager.create_sample_dataset()
    
    # 保存和加载数据集
    dataset_manager.save_dataset(sample_dataset, "sample_evaluation_dataset.json")
    loaded_dataset = dataset_manager.load_dataset("sample_evaluation_dataset.json")
    
    print(f"📊 数据集信息:")
    print(f"   案例数量: {len(loaded_dataset)}")
    for case in loaded_dataset:
        print(f"   - {case.id}: {case.question[:50]}... (难度: {case.difficulty}, 类别: {case.category})")
    
    return loaded_dataset

if __name__ == "__main__":
    test_benchmark_datasets()
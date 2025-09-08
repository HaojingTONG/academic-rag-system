# src/evaluation/evaluation_metrics.py
"""
RAG系统评估指标实现
包括上下文相关性、答案忠实度、答案相关性等核心指标
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util
import requests
import time

@dataclass 
class EvaluationResult:
    """评估结果数据结构"""
    context_relevance: float      # 上下文相关性 (0-1)
    faithfulness: float          # 答案忠实度 (0-1)
    answer_relevance: float      # 答案相关性 (0-1)
    context_precision: float     # 上下文精确度 (0-1)
    context_recall: float        # 上下文召回率 (0-1)
    overall_score: float         # 综合评分 (0-1)
    detailed_scores: Dict[str, Any]  # 详细分数
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'context_relevance': self.context_relevance,
            'faithfulness': self.faithfulness,
            'answer_relevance': self.answer_relevance,
            'context_precision': self.context_precision,
            'context_recall': self.context_recall,
            'overall_score': self.overall_score,
            'detailed_scores': self.detailed_scores
        }

class EvaluationMetrics:
    """RAG评估指标计算器"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        初始化评估指标计算器
        
        Args:
            embedding_model: 用于计算语义相似度的模型
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        print(f"✅ 加载评估用嵌入模型: {embedding_model}")
        
    def evaluate_single_case(self, 
                           question: str,
                           retrieved_contexts: List[str],
                           generated_answer: str,
                           ground_truth: Optional[str] = None,
                           relevant_contexts: Optional[List[str]] = None) -> EvaluationResult:
        """
        评估单个问答案例
        
        Args:
            question: 用户问题
            retrieved_contexts: 检索到的上下文列表
            generated_answer: 生成的答案
            ground_truth: 标准答案 (可选)
            relevant_contexts: 相关上下文列表 (用于计算召回率, 可选)
            
        Returns:
            EvaluationResult: 评估结果
        """
        print(f"📊 评估问题: {question[:50]}...")
        
        # 1. 计算上下文相关性
        context_relevance = self._calculate_context_relevance(question, retrieved_contexts)
        
        # 2. 计算答案忠实度  
        faithfulness = self._calculate_faithfulness(generated_answer, retrieved_contexts)
        
        # 3. 计算答案相关性
        answer_relevance = self._calculate_answer_relevance(question, generated_answer)
        
        # 4. 计算上下文精确度
        context_precision = self._calculate_context_precision(question, retrieved_contexts)
        
        # 5. 计算上下文召回率
        context_recall = self._calculate_context_recall(retrieved_contexts, relevant_contexts) if relevant_contexts else 0.0
        
        # 6. 计算综合评分
        overall_score = self._calculate_overall_score(
            context_relevance, faithfulness, answer_relevance, context_precision, context_recall
        )
        
        # 详细分数
        detailed_scores = {
            'semantic_similarity': self._calculate_semantic_similarity(question, generated_answer),
            'context_coverage': self._calculate_context_coverage(generated_answer, retrieved_contexts),
            'answer_length': len(generated_answer.split()),
            'context_count': len(retrieved_contexts),
            'avg_context_length': np.mean([len(ctx.split()) for ctx in retrieved_contexts]) if retrieved_contexts else 0
        }
        
        result = EvaluationResult(
            context_relevance=context_relevance,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            context_recall=context_recall,
            overall_score=overall_score,
            detailed_scores=detailed_scores
        )
        
        print(f"   ✅ 评估完成 - 综合评分: {overall_score:.3f}")
        return result
    
    def _calculate_context_relevance(self, question: str, contexts: List[str]) -> float:
        """计算上下文相关性 - 检索到的上下文与问题的相关程度"""
        if not contexts:
            return 0.0
            
        question_embedding = self.embedding_model.encode([question])
        context_embeddings = self.embedding_model.encode(contexts)
        
        similarities = util.cos_sim(question_embedding, context_embeddings)[0]
        
        # 计算平均相关性，但给高相关性更多权重
        similarities = similarities.numpy()
        weighted_relevance = np.mean(similarities) * 0.7 + np.max(similarities) * 0.3
        
        return float(np.clip(weighted_relevance, 0, 1))
    
    def _calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """计算答案忠实度 - 答案是否基于提供的上下文"""
        if not contexts or not answer:
            return 0.0
            
        # 将所有上下文合并
        combined_context = " ".join(contexts)
        
        # 计算语义相似度
        answer_embedding = self.embedding_model.encode([answer])
        context_embedding = self.embedding_model.encode([combined_context])
        
        semantic_similarity = util.cos_sim(answer_embedding, context_embedding)[0][0]
        
        # 计算词汇重叠度
        answer_words = set(answer.lower().split())
        context_words = set(combined_context.lower().split())
        
        if len(answer_words) == 0:
            lexical_overlap = 0.0
        else:
            lexical_overlap = len(answer_words.intersection(context_words)) / len(answer_words)
        
        # 综合语义和词汇相似度
        faithfulness = float(semantic_similarity * 0.7 + lexical_overlap * 0.3)
        
        return np.clip(faithfulness, 0, 1)
    
    def _calculate_answer_relevance(self, question: str, answer: str) -> float:
        """计算答案相关性 - 答案是否直接回答了问题"""
        if not question or not answer:
            return 0.0
            
        question_embedding = self.embedding_model.encode([question])
        answer_embedding = self.embedding_model.encode([answer])
        
        similarity = util.cos_sim(question_embedding, answer_embedding)[0][0]
        
        return float(np.clip(similarity, 0, 1))
    
    def _calculate_context_precision(self, question: str, contexts: List[str]) -> float:
        """计算上下文精确度 - 检索到的上下文中有多少是相关的"""
        if not contexts:
            return 0.0
            
        question_embedding = self.embedding_model.encode([question])
        context_embeddings = self.embedding_model.encode(contexts)
        
        similarities = util.cos_sim(question_embedding, context_embeddings)[0]
        
        # 设定相关性阈值
        relevance_threshold = 0.5
        relevant_count = sum(1 for sim in similarities if sim > relevance_threshold)
        
        precision = relevant_count / len(contexts)
        return float(precision)
    
    def _calculate_context_recall(self, retrieved_contexts: List[str], 
                                relevant_contexts: Optional[List[str]]) -> float:
        """计算上下文召回率 - 相关文档中有多少被检索到"""
        if not relevant_contexts or not retrieved_contexts:
            return 0.0
            
        retrieved_embeddings = self.embedding_model.encode(retrieved_contexts)
        relevant_embeddings = self.embedding_model.encode(relevant_contexts)
        
        # 计算检索到的上下文与相关上下文的相似度
        similarities = util.cos_sim(retrieved_embeddings, relevant_embeddings)
        
        # 对每个相关文档，检查是否有足够相似的检索结果
        similarity_threshold = 0.7
        recalled_count = 0
        
        for i in range(len(relevant_contexts)):
            max_similarity = np.max(similarities[:, i])
            if max_similarity > similarity_threshold:
                recalled_count += 1
                
        recall = recalled_count / len(relevant_contexts)
        return float(recall)
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的语义相似度"""
        if not text1 or not text2:
            return 0.0
            
        embeddings = self.embedding_model.encode([text1, text2])
        similarity = util.cos_sim(embeddings[0:1], embeddings[1:2])[0][0]
        
        return float(similarity)
    
    def _calculate_context_coverage(self, answer: str, contexts: List[str]) -> float:
        """计算答案对上下文的覆盖程度"""
        if not contexts or not answer:
            return 0.0
            
        answer_words = set(answer.lower().split())
        context_words = set()
        
        for context in contexts:
            context_words.update(context.lower().split())
            
        if len(context_words) == 0:
            return 0.0
            
        coverage = len(answer_words.intersection(context_words)) / len(context_words)
        return float(np.clip(coverage, 0, 1))
    
    def _calculate_overall_score(self, context_relevance: float, faithfulness: float,
                               answer_relevance: float, context_precision: float,
                               context_recall: float) -> float:
        """计算综合评分"""
        # 权重设置
        weights = {
            'context_relevance': 0.25,
            'faithfulness': 0.30,
            'answer_relevance': 0.25, 
            'context_precision': 0.15,
            'context_recall': 0.05  # 召回率权重较低，因为通常没有完整的相关文档集
        }
        
        overall_score = (
            context_relevance * weights['context_relevance'] +
            faithfulness * weights['faithfulness'] +
            answer_relevance * weights['answer_relevance'] +
            context_precision * weights['context_precision'] +
            context_recall * weights['context_recall']
        )
        
        return float(np.clip(overall_score, 0, 1))

class LLMBasedEvaluator:
    """基于LLM的评估器 - 使用大语言模型进行更深入的评估"""
    
    def __init__(self, llm_base_url: str = "http://localhost:11434", model: str = "llama3.1:8b"):
        """
        初始化LLM评估器
        
        Args:
            llm_base_url: LLM服务的基础URL
            model: 使用的模型名称
        """
        self.base_url = llm_base_url
        self.model = model
        print(f"✅ 初始化LLM评估器: {model}")
        
    def evaluate_with_llm(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """使用LLM评估问答质量"""
        
        evaluation_prompt = f"""
请作为一个专业的RAG系统评估专家，从以下几个维度评估这个问答案例的质量：

问题: {question}

上下文: {context}

生成的答案: {answer}

请从以下维度进行评估（1-5分，5分最好）：

1. 上下文相关性: 提供的上下文与问题的相关程度
2. 答案忠实度: 答案是否完全基于提供的上下文，没有虚构信息  
3. 答案相关性: 答案是否直接回答了用户的问题
4. 答案完整性: 答案是否完整，有没有遗漏重要信息
5. 答案清晰度: 答案是否表达清晰，易于理解

请以JSON格式返回评估结果：
{{
  "context_relevance": <分数>,
  "faithfulness": <分数>,
  "answer_relevance": <分数>, 
  "completeness": <分数>,
  "clarity": <分数>,
  "reasoning": "<详细的评估理由>"
}}
"""
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": evaluation_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer_text = result.get('response', '')
                
                # 尝试解析JSON结果
                try:
                    # 提取JSON部分
                    json_start = answer_text.find('{')
                    json_end = answer_text.rfind('}') + 1
                    if json_start != -1 and json_end != 0:
                        json_str = answer_text[json_start:json_end]
                        evaluation_result = json.loads(json_str)
                        
                        # 标准化评分 (1-5 -> 0-1)
                        for key in ['context_relevance', 'faithfulness', 'answer_relevance', 'completeness', 'clarity']:
                            if key in evaluation_result:
                                evaluation_result[key] = (evaluation_result[key] - 1) / 4
                                
                        return evaluation_result
                except json.JSONDecodeError:
                    pass
                    
            return {"error": f"LLM评估失败: {response.status_code}"}
            
        except Exception as e:
            return {"error": f"LLM评估异常: {str(e)}"}

# 使用示例和测试函数
def test_evaluation_metrics():
    """测试评估指标"""
    print("🧪 测试评估指标...")
    
    evaluator = EvaluationMetrics()
    
    # 测试用例
    question = "What is attention mechanism in neural networks?"
    contexts = [
        "Attention mechanism allows neural networks to focus on relevant parts of input sequences.",
        "The attention mechanism was introduced in the Transformer architecture.",
        "Attention computes weighted averages of input representations."
    ]
    answer = "Attention mechanism is a technique that allows neural networks to selectively focus on relevant parts of the input sequence when making predictions."
    
    result = evaluator.evaluate_single_case(question, contexts, answer)
    
    print(f"📊 评估结果:")
    print(f"   上下文相关性: {result.context_relevance:.3f}")
    print(f"   答案忠实度: {result.faithfulness:.3f}") 
    print(f"   答案相关性: {result.answer_relevance:.3f}")
    print(f"   综合评分: {result.overall_score:.3f}")
    
    return result

if __name__ == "__main__":
    test_evaluation_metrics()
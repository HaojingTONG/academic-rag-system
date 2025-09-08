# src/evaluation/rag_evaluator.py
"""
RAG系统完整评估器
整合各种评估指标和数据集，提供完整的RAG系统评估功能
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

from .evaluation_metrics import EvaluationMetrics, EvaluationResult, LLMBasedEvaluator
from .benchmark_datasets import BenchmarkDatasets, EvaluationCase

class RAGEvaluator:
    """RAG系统评估器主类"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_base_url: str = "http://localhost:11434",
                 llm_model: str = "llama3.1:8b",
                 output_dir: str = "data/evaluation/results"):
        """
        初始化RAG评估器
        
        Args:
            embedding_model: 用于语义相似度计算的嵌入模型
            llm_base_url: LLM服务URL
            llm_model: LLM模型名称
            output_dir: 评估结果输出目录
        """
        self.metrics = EvaluationMetrics(embedding_model)
        self.llm_evaluator = LLMBasedEvaluator(llm_base_url, llm_model)
        self.dataset_manager = BenchmarkDatasets()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"✅ RAG评估器初始化完成")
        print(f"   嵌入模型: {embedding_model}")
        print(f"   LLM模型: {llm_model}")
        print(f"   输出目录: {output_dir}")
    
    def evaluate_rag_system(self, 
                           rag_system,
                           evaluation_dataset: List[EvaluationCase],
                           use_llm_evaluation: bool = True,
                           save_results: bool = True) -> Dict[str, Any]:
        """
        完整评估RAG系统
        
        Args:
            rag_system: RAG系统实例，需要有query方法
            evaluation_dataset: 评估数据集
            use_llm_evaluation: 是否使用LLM进行额外评估
            save_results: 是否保存评估结果
            
        Returns:
            Dict: 评估结果报告
        """
        print(f"🚀 开始评估RAG系统 ({len(evaluation_dataset)} 个测试案例)...")
        print("=" * 80)
        
        evaluation_results = []
        start_time = time.time()
        
        for i, case in enumerate(evaluation_dataset):
            print(f"\n📋 案例 [{i+1}/{len(evaluation_dataset)}]: {case.question[:60]}...")
            print(f"   难度: {case.difficulty} | 类别: {case.category}")
            
            # 使用RAG系统生成答案
            try:
                rag_response = rag_system.query(case.question)
                
                # 提取RAG系统返回的信息
                if isinstance(rag_response, dict):
                    generated_answer = rag_response.get('answer', '')
                    retrieved_contexts = rag_response.get('contexts', [])
                elif isinstance(rag_response, str):
                    generated_answer = rag_response
                    retrieved_contexts = []  # 无法获取检索上下文
                else:
                    print(f"   ⚠️ 无法解析RAG系统响应: {type(rag_response)}")
                    continue
                    
            except Exception as e:
                print(f"   ❌ RAG系统查询失败: {e}")
                continue
            
            # 核心指标评估
            metric_result = self.metrics.evaluate_single_case(
                question=case.question,
                retrieved_contexts=retrieved_contexts,
                generated_answer=generated_answer,
                ground_truth=case.ground_truth_answer,
                relevant_contexts=case.relevant_contexts
            )
            
            # LLM评估（可选）
            llm_result = {}
            if use_llm_evaluation and retrieved_contexts:
                combined_context = "\n".join(retrieved_contexts)
                llm_result = self.llm_evaluator.evaluate_with_llm(
                    case.question, combined_context, generated_answer
                )
            
            # 整合结果
            case_result = {
                'case_id': case.id,
                'question': case.question,
                'difficulty': case.difficulty,
                'category': case.category,
                'generated_answer': generated_answer,
                'ground_truth_answer': case.ground_truth_answer,
                'retrieved_contexts': retrieved_contexts,
                'metric_scores': metric_result.to_dict(),
                'llm_scores': llm_result,
                'metadata': case.metadata
            }
            
            evaluation_results.append(case_result)
            
            print(f"   ✅ 评估完成 - 综合评分: {metric_result.overall_score:.3f}")
            
            # 避免请求过于频繁
            if use_llm_evaluation:
                time.sleep(0.5)
        
        # 计算总体统计
        evaluation_time = time.time() - start_time
        summary_report = self._generate_summary_report(evaluation_results, evaluation_time)
        
        # 保存结果
        if save_results:
            self._save_evaluation_results(evaluation_results, summary_report)
        
        print("\n" + "=" * 80)
        print("📊 评估完成!")
        self._print_summary_report(summary_report)
        
        return {
            'summary': summary_report,
            'detailed_results': evaluation_results
        }
    
    def _generate_summary_report(self, results: List[Dict], evaluation_time: float) -> Dict[str, Any]:
        """生成评估摘要报告"""
        if not results:
            return {}
            
        # 提取所有指标分数
        all_scores = []
        for result in results:
            if 'metric_scores' in result:
                all_scores.append(result['metric_scores'])
        
        if not all_scores:
            return {}
        
        # 计算平均分数
        avg_scores = {}
        score_keys = ['context_relevance', 'faithfulness', 'answer_relevance', 
                     'context_precision', 'context_recall', 'overall_score']
        
        for key in score_keys:
            scores = [score.get(key, 0) for score in all_scores if key in score]
            if scores:
                avg_scores[f'avg_{key}'] = np.mean(scores)
                avg_scores[f'std_{key}'] = np.std(scores)
                avg_scores[f'min_{key}'] = np.min(scores)
                avg_scores[f'max_{key}'] = np.max(scores)
        
        # 按难度和类别分组统计
        difficulty_stats = self._calculate_group_stats(results, 'difficulty')
        category_stats = self._calculate_group_stats(results, 'category')
        
        # 生成报告
        summary = {
            'evaluation_info': {
                'total_cases': len(results),
                'evaluation_time': evaluation_time,
                'timestamp': datetime.now().isoformat()
            },
            'overall_performance': avg_scores,
            'difficulty_breakdown': difficulty_stats,
            'category_breakdown': category_stats,
            'top_performing_cases': self._get_top_cases(results, 'best'),
            'worst_performing_cases': self._get_top_cases(results, 'worst')
        }
        
        return summary
    
    def _calculate_group_stats(self, results: List[Dict], group_key: str) -> Dict[str, Any]:
        """计算分组统计信息"""
        groups = {}
        
        for result in results:
            group_value = result.get(group_key, 'unknown')
            if group_value not in groups:
                groups[group_value] = []
            
            if 'metric_scores' in result:
                overall_score = result['metric_scores'].get('overall_score', 0)
                groups[group_value].append(overall_score)
        
        group_stats = {}
        for group, scores in groups.items():
            if scores:
                group_stats[group] = {
                    'count': len(scores),
                    'avg_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores)
                }
        
        return group_stats
    
    def _get_top_cases(self, results: List[Dict], mode: str = 'best', n: int = 3) -> List[Dict]:
        """获取表现最好/最差的案例"""
        scored_cases = []
        
        for result in results:
            if 'metric_scores' in result:
                overall_score = result['metric_scores'].get('overall_score', 0)
                scored_cases.append({
                    'case_id': result['case_id'],
                    'question': result['question'][:100] + '...' if len(result['question']) > 100 else result['question'],
                    'overall_score': overall_score,
                    'difficulty': result.get('difficulty', 'unknown'),
                    'category': result.get('category', 'unknown')
                })
        
        # 排序
        reverse = (mode == 'best')
        scored_cases.sort(key=lambda x: x['overall_score'], reverse=reverse)
        
        return scored_cases[:n]
    
    def _save_evaluation_results(self, results: List[Dict], summary: Dict[str, Any]):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        detailed_file = self.output_dir / f"evaluation_detailed_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存摘要报告
        summary_file = self.output_dir / f"evaluation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 生成CSV报告
        self._generate_csv_report(results, timestamp)
        
        # 生成可视化图表
        self._generate_visualizations(results, summary, timestamp)
        
        print(f"💾 评估结果已保存:")
        print(f"   详细结果: {detailed_file}")
        print(f"   摘要报告: {summary_file}")
    
    def _generate_csv_report(self, results: List[Dict], timestamp: str):
        """生成CSV格式的评估报告"""
        csv_data = []
        
        for result in results:
            row = {
                'case_id': result['case_id'],
                'difficulty': result.get('difficulty', ''),
                'category': result.get('category', ''),
                'question_length': len(result['question']),
                'answer_length': len(result.get('generated_answer', '')),
                'context_count': len(result.get('retrieved_contexts', []))
            }
            
            # 添加评估指标
            if 'metric_scores' in result:
                metrics = result['metric_scores']
                row.update({
                    'context_relevance': metrics.get('context_relevance', 0),
                    'faithfulness': metrics.get('faithfulness', 0),
                    'answer_relevance': metrics.get('answer_relevance', 0),
                    'context_precision': metrics.get('context_precision', 0),
                    'context_recall': metrics.get('context_recall', 0),
                    'overall_score': metrics.get('overall_score', 0)
                })
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / f"evaluation_report_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"   CSV报告: {csv_file}")
    
    def _generate_visualizations(self, results: List[Dict], summary: Dict[str, Any], timestamp: str):
        """生成评估结果可视化图表"""
        try:
            # 准备数据
            scores_data = []
            for result in results:
                if 'metric_scores' in result:
                    metrics = result['metric_scores']
                    scores_data.append({
                        'difficulty': result.get('difficulty', 'unknown'),
                        'category': result.get('category', 'unknown'),
                        'context_relevance': metrics.get('context_relevance', 0),
                        'faithfulness': metrics.get('faithfulness', 0),
                        'answer_relevance': metrics.get('answer_relevance', 0),
                        'overall_score': metrics.get('overall_score', 0)
                    })
            
            if not scores_data:
                return
                
            df = pd.DataFrame(scores_data)
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('RAG System Evaluation Results', fontsize=16)
            
            # 1. 整体分数分布
            axes[0, 0].hist(df['overall_score'], bins=20, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Overall Score Distribution')
            axes[0, 0].set_xlabel('Overall Score')
            axes[0, 0].set_ylabel('Frequency')
            
            # 2. 按难度分组的平均分数
            if 'difficulty' in df.columns:
                difficulty_scores = df.groupby('difficulty')['overall_score'].mean()
                axes[0, 1].bar(difficulty_scores.index, difficulty_scores.values, color='lightcoral')
                axes[0, 1].set_title('Average Score by Difficulty')
                axes[0, 1].set_xlabel('Difficulty')
                axes[0, 1].set_ylabel('Average Score')
            
            # 3. 各指标对比
            metric_cols = ['context_relevance', 'faithfulness', 'answer_relevance']
            metric_means = [df[col].mean() for col in metric_cols]
            axes[1, 0].bar(metric_cols, metric_means, color='lightgreen')
            axes[1, 0].set_title('Average Scores by Metric')
            axes[1, 0].set_ylabel('Average Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 4. 各指标热力图
            if len(df) > 1:
                correlation_matrix = df[metric_cols + ['overall_score']].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
                axes[1, 1].set_title('Metric Correlations')
            
            plt.tight_layout()
            
            # 保存图表
            chart_file = self.output_dir / f"evaluation_charts_{timestamp}.png"
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   可视化图表: {chart_file}")
            
        except Exception as e:
            print(f"   ⚠️ 图表生成失败: {e}")
    
    def _print_summary_report(self, summary: Dict[str, Any]):
        """打印评估摘要报告"""
        if not summary:
            return
            
        info = summary.get('evaluation_info', {})
        performance = summary.get('overall_performance', {})
        
        print(f"📊 评估摘要:")
        print(f"   总案例数: {info.get('total_cases', 0)}")
        print(f"   评估耗时: {info.get('evaluation_time', 0):.1f}秒")
        print()
        
        print(f"🎯 整体性能:")
        if 'avg_overall_score' in performance:
            print(f"   综合评分: {performance['avg_overall_score']:.3f} ± {performance.get('std_overall_score', 0):.3f}")
            print(f"   上下文相关性: {performance.get('avg_context_relevance', 0):.3f}")
            print(f"   答案忠实度: {performance.get('avg_faithfulness', 0):.3f}")
            print(f"   答案相关性: {performance.get('avg_answer_relevance', 0):.3f}")
        print()
        
        # 按难度统计
        difficulty_stats = summary.get('difficulty_breakdown', {})
        if difficulty_stats:
            print(f"📈 按难度分组:")
            for difficulty, stats in difficulty_stats.items():
                print(f"   {difficulty}: {stats['avg_score']:.3f} ({stats['count']} 案例)")
        print()
        
        # 最佳/最差案例
        best_cases = summary.get('top_performing_cases', [])
        if best_cases:
            print(f"🏆 表现最佳案例:")
            for case in best_cases:
                print(f"   - {case['case_id']}: {case['overall_score']:.3f}")
        print()

# 使用示例
def test_rag_evaluator():
    """测试RAG评估器"""
    print("🧪 测试RAG评估器...")
    
    # 模拟RAG系统
    class MockRAGSystem:
        def query(self, question: str) -> Dict[str, Any]:
            return {
                'answer': f"This is a sample answer for: {question}",
                'contexts': [
                    "Sample context 1 related to the question.",
                    "Sample context 2 providing additional information."
                ]
            }
    
    # 初始化评估器
    evaluator = RAGEvaluator()
    
    # 创建测试数据集
    dataset_manager = BenchmarkDatasets()
    test_dataset = dataset_manager.create_sample_dataset()
    
    # 评估RAG系统
    mock_rag = MockRAGSystem()
    results = evaluator.evaluate_rag_system(
        rag_system=mock_rag,
        evaluation_dataset=test_dataset,
        use_llm_evaluation=False  # 测试时关闭LLM评估
    )
    
    return results

if __name__ == "__main__":
    test_rag_evaluator()
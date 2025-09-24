#!/usr/bin/env python3
"""
RAG系统评估脚本
使用评估框架对学术论文RAG系统进行完整评估
"""

import sys
from pathlib import Path
import json

# 添加src到路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'src'))

from src.evaluation import RAGEvaluator, BenchmarkDatasets
from main_rag_system import MainRAGSystem as AcademicRAGSystem

def main():
    """主评估流程"""
    print("🚀 RAG系统评估开始")
    print("=" * 80)
    
    # 1. 初始化RAG系统
    print("📚 初始化RAG系统...")
    rag_system = AcademicRAGSystem()
    
    # 2. 初始化评估器
    print("📊 初始化评估器...")
    evaluator = RAGEvaluator(
        embedding_model="all-MiniLM-L6-v2",
        llm_base_url="http://localhost:11434",
        llm_model="llama3.1:8b",
        output_dir="data/evaluation/results"
    )
    
    # 3. 准备评估数据集
    print("📋 准备评估数据集...")
    dataset_manager = BenchmarkDatasets("data/evaluation")
    
    # 选择评估数据集创建方式
    choice = input("选择数据集创建方式:\n1. 使用示例数据集\n2. 基于论文数据生成新数据集\n3. 加载已有数据集\n请输入选择 (1-3): ").strip()
    
    if choice == '1':
        # 使用示例数据集
        print("📄 使用示例数据集...")
        evaluation_dataset = dataset_manager.create_sample_dataset()
        
    elif choice == '2':
        # 基于论文数据生成数据集
        print("🔄 基于论文数据生成评估数据集...")
        papers_data_path = "data/main_system_papers.json"
        
        if not Path(papers_data_path).exists():
            print(f"❌ 论文数据文件不存在: {papers_data_path}")
            print("请先运行PDF处理脚本生成论文数据")
            return
            
        num_questions = int(input("输入要生成的问题数量 (推荐20-50): ") or "20")
        
        evaluation_dataset = dataset_manager.generate_academic_qa_dataset(
            papers_data_path=papers_data_path,
            num_questions=num_questions,
            llm_base_url="http://localhost:11434",
            model="llama3.1:8b"
        )
        
        # 保存生成的数据集
        if evaluation_dataset:
            dataset_filename = f"academic_qa_dataset_{len(evaluation_dataset)}.json"
            dataset_manager.save_dataset(evaluation_dataset, dataset_filename)
            print(f"✅ 数据集已保存: {dataset_filename}")
        
    elif choice == '3':
        # 加载已有数据集
        print("📂 加载已有数据集...")
        dataset_files = list(Path("data/evaluation").glob("*.json"))
        
        if not dataset_files:
            print("❌ 未找到已有数据集，使用示例数据集")
            evaluation_dataset = dataset_manager.create_sample_dataset()
        else:
            print("可用数据集:")
            for i, file in enumerate(dataset_files, 1):
                print(f"   {i}. {file.name}")
            
            file_choice = int(input(f"选择数据集 (1-{len(dataset_files)}): ") or "1") - 1
            selected_file = dataset_files[file_choice]
            evaluation_dataset = dataset_manager.load_dataset(selected_file.name)
    
    else:
        print("❌ 无效选择，使用示例数据集")
        evaluation_dataset = dataset_manager.create_sample_dataset()
    
    if not evaluation_dataset:
        print("❌ 无法创建评估数据集")
        return
    
    print(f"✅ 评估数据集准备完成: {len(evaluation_dataset)} 个案例")
    
    # 4. 配置评估选项
    print("\n📋 评估配置:")
    use_llm_eval = input("是否启用LLM深度评估? (y/n, 默认n): ").lower().startswith('y')
    
    if use_llm_eval:
        print("⚠️ 注意: LLM评估会显著增加评估时间")
    
    # 5. 开始评估
    print(f"\n🎯 开始评估RAG系统...")
    print(f"   数据集: {len(evaluation_dataset)} 个案例")
    print(f"   LLM评估: {'启用' if use_llm_eval else '禁用'}")
    print("=" * 80)
    
    try:
        # 适配RAG系统的query方法
        class RAGSystemAdapter:
            def __init__(self, rag_system):
                self.rag_system = rag_system
                
            def query(self, question: str):
                """适配器方法，确保返回正确格式"""
                try:
                    # 调用RAG系统
                    response = self.rag_system.query_academic_papers(question)
                    
                    # 提取答案和上下文
                    if isinstance(response, dict):
                        answer = response.get('answer', '')
                        contexts = []
                        
                        # 从final_results中提取上下文
                        final_results = response.get('final_results', [])
                        for result in final_results:
                            if isinstance(result, dict) and 'text' in result:
                                contexts.append(result['text'])
                            elif isinstance(result, str):
                                contexts.append(result)
                        
                        return {
                            'answer': answer,
                            'contexts': contexts
                        }
                    else:
                        return {
                            'answer': str(response),
                            'contexts': []
                        }
                        
                except Exception as e:
                    print(f"   ⚠️ RAG查询失败: {e}")
                    return {
                        'answer': f"查询失败: {e}",
                        'contexts': []
                    }
        
        # 创建适配器
        rag_adapter = RAGSystemAdapter(rag_system)
        
        # 执行评估
        results = evaluator.evaluate_rag_system(
            rag_system=rag_adapter,
            evaluation_dataset=evaluation_dataset,
            use_llm_evaluation=use_llm_eval,
            save_results=True
        )
        
        print("\n🎉 评估完成!")
        print("=" * 80)
        
        # 显示简要结果
        summary = results.get('summary', {})
        performance = summary.get('overall_performance', {})
        
        if 'avg_overall_score' in performance:
            print(f"📊 RAG系统评估结果:")
            print(f"   🎯 综合评分: {performance['avg_overall_score']:.3f}/1.0")
            print(f"   🔍 上下文相关性: {performance.get('avg_context_relevance', 0):.3f}/1.0")
            print(f"   🤝 答案忠实度: {performance.get('avg_faithfulness', 0):.3f}/1.0")
            print(f"   💡 答案相关性: {performance.get('avg_answer_relevance', 0):.3f}/1.0")
            print(f"   📈 上下文精确度: {performance.get('avg_context_precision', 0):.3f}/1.0")
            
            # 评估等级
            overall_score = performance['avg_overall_score']
            if overall_score >= 0.8:
                grade = "优秀 🏆"
            elif overall_score >= 0.7:
                grade = "良好 👍"
            elif overall_score >= 0.6:
                grade = "中等 👌"
            elif overall_score >= 0.5:
                grade = "及格 📝"
            else:
                grade = "需要改进 📉"
                
            print(f"   📋 评估等级: {grade}")
        
        # 改进建议
        print(f"\n💡 改进建议:")
        if performance.get('avg_context_relevance', 0) < 0.7:
            print("   - 优化检索算法，提高上下文相关性")
        if performance.get('avg_faithfulness', 0) < 0.7:
            print("   - 改进提示工程，确保答案基于上下文")
        if performance.get('avg_answer_relevance', 0) < 0.7:
            print("   - 优化问答匹配，提高答案相关性")
        if performance.get('avg_context_precision', 0) < 0.6:
            print("   - 提高检索精确度，过滤无关内容")
            
        print(f"\n📁 详细结果已保存到: data/evaluation/results/")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 评估被用户中断")
    except Exception as e:
        print(f"\n❌ 评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

def quick_evaluation():
    """快速评估模式"""
    print("⚡ 快速评估模式")
    print("=" * 50)
    
    # 使用简化配置
    rag_system = AcademicRAGSystem()
    evaluator = RAGEvaluator(output_dir="data/evaluation/quick_results")
    dataset_manager = BenchmarkDatasets()
    
    # 使用示例数据集
    evaluation_dataset = dataset_manager.create_sample_dataset()
    
    class QuickRAGAdapter:
        def __init__(self, rag_system):
            self.rag_system = rag_system
            
        def query(self, question: str):
            try:
                response = self.rag_system.query_academic_papers(question)
                return {
                    'answer': response.get('answer', '') if isinstance(response, dict) else str(response),
                    'contexts': ['Sample context for quick evaluation']
                }
            except:
                return {'answer': 'Quick evaluation answer', 'contexts': []}
    
    rag_adapter = QuickRAGAdapter(rag_system)
    
    results = evaluator.evaluate_rag_system(
        rag_system=rag_adapter,
        evaluation_dataset=evaluation_dataset,
        use_llm_evaluation=False,
        save_results=True
    )
    
    print("⚡ 快速评估完成!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_evaluation()
    else:
        main()
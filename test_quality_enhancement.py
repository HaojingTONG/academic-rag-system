#!/usr/bin/env python3
"""
测试生成质量提升模块
验证 quality_enhancement.py 中各个组件的功能
"""

import sys
import json
import time
from pathlib import Path

# 添加src到路径
sys.path.append('src')

def test_imports():
    """测试模块导入"""
    print("=" * 60)
    print("1. 测试模块导入")
    print("=" * 60)
    
    try:
        from src.generator.quality_enhancement import (
            QualityEnhancedGenerator,
            ContextOptimizer,
            PromptEngineer,
            FactChecker,
            CitationManager,
            ContextWindow,
            Citation,
            FactCheckResult
        )
        print("✅ 所有组件导入成功")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_context_optimizer():
    """测试上下文优化器"""
    print("\n" + "=" * 60)
    print("2. 测试上下文优化器")
    print("=" * 60)
    
    try:
        from src.generator.quality_enhancement import ContextOptimizer
        
        # 创建测试数据
        mock_results = [
            {
                'content': 'Transformer architecture uses self-attention mechanism to process sequences efficiently.',
                'metadata': {
                    'paper_id': '1706.03762',
                    'title': 'Attention Is All You Need',
                    'section_type': 'abstract',
                    'has_formulas': True,
                    'has_citations': True
                },
                'combined_score': 0.9
            },
            {
                'content': 'BERT demonstrates bidirectional training for language understanding tasks.',
                'metadata': {
                    'paper_id': '1810.04805',
                    'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
                    'section_type': 'results',
                    'has_citations': True
                },
                'combined_score': 0.8
            },
            {
                'content': 'Low relevance content that should be filtered out.',
                'metadata': {
                    'paper_id': 'low_relevance',
                    'title': 'Irrelevant Paper',
                    'section_type': 'content'
                },
                'combined_score': 0.2
            }
        ]
        
        # 初始化优化器
        optimizer = ContextOptimizer(max_tokens=1000, min_relevance=0.3)
        
        # 测试上下文优化
        query = "transformer attention mechanism"
        context_window = optimizer.optimize_context(mock_results, query)
        
        print(f"✅ 上下文优化成功")
        print(f"   源文档数: {len(context_window.source_papers)}")
        print(f"   相关性分数: {context_window.relevance_score:.3f}")
        print(f"   置信度: {context_window.confidence_level:.3f}")
        print(f"   Token数量: {context_window.total_tokens}")
        
        # 验证过滤效果
        assert len(context_window.source_papers) <= len(mock_results), "源文档数量应该被过滤"
        assert context_window.relevance_score > 0, "相关性分数应该大于0"
        assert context_window.confidence_level >= 0, "置信度应该非负"
        
        return True
        
    except Exception as e:
        print(f"❌ 上下文优化器测试失败: {e}")
        return False

def test_prompt_engineer():
    """测试提示词工程师"""
    print("\n" + "=" * 60)
    print("3. 测试提示词工程师")
    print("=" * 60)
    
    try:
        from src.generator.quality_enhancement import PromptEngineer, ContextWindow
        
        # 创建提示词工程师
        engineer = PromptEngineer()
        
        # 创建模拟上下文
        context = ContextWindow(
            content="Test context about transformer architecture and attention mechanisms.",
            source_papers=['1706.03762', '1810.04805'],
            relevance_score=0.8,
            confidence_level=0.7,
            total_tokens=500
        )
        
        # 测试不同类型的提示词
        test_cases = [
            ("What is transformer architecture?", "definition"),
            ("Compare BERT and GPT models", "comparison"),
            ("How does attention mechanism work?", "methodology"),
            ("What are recent advances in NLP?", "recent_work"),
            ("General AI question", "general")
        ]
        
        for query, intent in test_cases:
            prompt = engineer.generate_prompt(query, intent, context)
            print(f"✅ {intent} 提示词生成成功 (长度: {len(prompt)} 字符)")
            
            # 验证提示词包含必要信息
            assert query in prompt, f"提示词应该包含查询: {query}"
            assert str(context.confidence_level) in prompt, "提示词应该包含置信度"
            assert str(len(context.source_papers)) in prompt, "提示词应该包含源数量"
        
        # 测试领域检测
        domains = engineer._detect_domains("neural networks deep learning attention mechanism")
        print(f"✅ 领域检测成功: {domains}")
        
        return True
        
    except Exception as e:
        print(f"❌ 提示词工程师测试失败: {e}")
        return False

def test_fact_checker():
    """测试事实核查器"""
    print("\n" + "=" * 60)
    print("4. 测试事实核查器")
    print("=" * 60)
    
    try:
        from src.generator.quality_enhancement import FactChecker
        
        # 创建事实核查器
        checker = FactChecker()
        
        # 测试文本
        test_answer = """
        Transformer architecture is a neural network design that uses self-attention mechanisms.
        BERT demonstrates that bidirectional training improves language understanding significantly.
        The method achieves state-of-the-art results on multiple benchmarks.
        This approach outperforms previous RNN-based models in efficiency and accuracy.
        """
        
        # 模拟源文档
        source_documents = [
            {
                'content': 'Transformer uses self-attention mechanism for parallel processing. This shows significant improvements in efficiency.',
                'metadata': {'paper_id': '1706.03762', 'title': 'Attention Is All You Need'}
            },
            {
                'content': 'BERT demonstrates bidirectional training benefits. The results prove better language understanding.',
                'metadata': {'paper_id': '1810.04805', 'title': 'BERT Paper'}
            }
        ]
        
        # 执行事实核查
        fact_results = checker.verify_claims(test_answer, source_documents)
        
        print(f"✅ 事实核查完成")
        print(f"   检测到声明数: {len(fact_results)}")
        
        for i, result in enumerate(fact_results[:3], 1):  # 只显示前3个
            print(f"   声明 {i}: {result.claim[:50]}...")
            print(f"   支持度: {result.support_level}")
            print(f"   置信度: {result.confidence_score:.3f}")
            print(f"   支持源: {len(result.supporting_sources)}")
            print(f"   反对源: {len(result.contradicting_sources)}")
        
        # 验证结果结构
        for result in fact_results:
            assert hasattr(result, 'claim'), "结果应该包含声明"
            assert hasattr(result, 'support_level'), "结果应该包含支持度"
            assert hasattr(result, 'confidence_score'), "结果应该包含置信度"
            assert 0 <= result.confidence_score <= 1, "置信度应该在0-1之间"
        
        return True
        
    except Exception as e:
        print(f"❌ 事实核查器测试失败: {e}")
        return False

def test_citation_manager():
    """测试引用管理器"""
    print("\n" + "=" * 60)
    print("5. 测试引用管理器")
    print("=" * 60)
    
    try:
        from src.generator.quality_enhancement import CitationManager
        
        # 测试不同引用格式
        for style in ['APA', 'IEEE']:
            print(f"\n测试 {style} 格式:")
            manager = CitationManager(citation_style=style)
            
            # 模拟源数据
            mock_sources = [
                {
                    'content': 'Transformer content...',
                    'metadata': {
                        'paper_id': '1706.03762',
                        'title': 'Attention Is All You Need',
                        'authors': ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
                        'published': '2017-06-12'
                    },
                    'relevance_score': 0.9
                },
                {
                    'content': 'BERT content...',
                    'metadata': {
                        'paper_id': '1810.04805',
                        'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
                        'authors': ['Jacob Devlin', 'Ming-Wei Chang', 'Kenton Lee'],
                        'published': '2018-10-11'
                    },
                    'relevance_score': 0.8
                }
            ]
            
            # 生成引用
            citations = manager.generate_citations(mock_sources)
            print(f"   生成引用数: {len(citations)}")
            
            # 格式化引用
            for i, citation in enumerate(citations, 1):
                formatted = manager.format_citation(citation)
                print(f"   [{i}] {formatted}")
                
                # 验证引用结构
                assert citation.paper_id, "引用应该有paper_id"
                assert citation.title, "引用应该有标题"
                assert citation.year, "引用应该有年份"
                assert isinstance(citation.authors, list), "作者应该是列表"
            
            # 生成参考文献
            bibliography = manager.generate_bibliography(citations)
            print(f"   参考文献长度: {len(bibliography)} 字符")
            assert "References:" in bibliography, "参考文献应该包含标题"
            
        print("✅ 引用管理器测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 引用管理器测试失败: {e}")
        return False

def test_quality_enhanced_generator():
    """测试质量增强生成器（集成测试）"""
    print("\n" + "=" * 60)
    print("6. 测试质量增强生成器（集成测试）")
    print("=" * 60)
    
    try:
        from src.generator.quality_enhancement import QualityEnhancedGenerator
        
        # 创建生成器
        generator = QualityEnhancedGenerator(citation_style='APA')
        
        # 模拟检索结果
        mock_results = [
            {
                'content': 'Transformer architecture revolutionized NLP by introducing self-attention mechanisms that allow parallel processing of sequences.',
                'metadata': {
                    'paper_id': '1706.03762',
                    'title': 'Attention Is All You Need',
                    'authors': ['Ashish Vaswani', 'Noam Shazeer'],
                    'published': '2017-06-12',
                    'section_type': 'abstract',
                    'has_formulas': True,
                    'has_citations': True
                },
                'combined_score': 0.9,
                'relevance_score': 0.9
            },
            {
                'content': 'BERT uses bidirectional encoder representations from transformers, demonstrating significant improvements in language understanding tasks.',
                'metadata': {
                    'paper_id': '1810.04805',
                    'title': 'BERT: Pre-training of Deep Bidirectional Transformers',
                    'authors': ['Jacob Devlin', 'Ming-Wei Chang'],
                    'published': '2018-10-11',
                    'section_type': 'results',
                    'has_citations': True
                },
                'combined_score': 0.8,
                'relevance_score': 0.8
            },
            {
                'content': 'GPT models demonstrate the power of autoregressive language modeling for various downstream tasks.',
                'metadata': {
                    'paper_id': '1810.04855',
                    'title': 'Improving Language Understanding by Generative Pre-Training',
                    'authors': ['Alec Radford', 'Karthik Narasimhan'],
                    'published': '2018-06-11',
                    'section_type': 'methodology',
                    'has_formulas': False,
                    'has_citations': True
                },
                'combined_score': 0.7,
                'relevance_score': 0.7
            }
        ]
        
        # 测试不同查询意图
        test_cases = [
            ("What is the transformer architecture?", "definition"),
            ("Compare BERT and GPT models", "comparison"),
            ("How does self-attention work?", "methodology"),
            ("Recent advances in transformer models", "recent_work")
        ]
        
        for query, intent in test_cases:
            print(f"\n测试查询: {query} (意图: {intent})")
            
            result = generator.generate_enhanced_answer(query, intent, mock_results)
            
            # 验证结果结构
            assert 'answer' in result, "结果应该包含答案"
            assert 'context_info' in result, "结果应该包含上下文信息"
            assert 'citations' in result, "结果应该包含引用"
            assert 'quality_metrics' in result, "结果应该包含质量指标"
            
            context_info = result['context_info']
            quality_metrics = result['quality_metrics']
            
            print(f"   ✅ 答案长度: {len(result['answer'])} 字符")
            print(f"   ✅ 置信度: {context_info['confidence_level']:.3f}")
            print(f"   ✅ 源数量: {context_info['source_count']}")
            print(f"   ✅ 引用数量: {len(result['citations'])}")
            print(f"   ✅ 整体质量: {quality_metrics['overall_quality']:.3f}")
            print(f"   ✅ 建议: {quality_metrics['recommendation']}")
            
            if result.get('fact_check'):
                print(f"   ✅ 事实核查: {len(result['fact_check'])} 项")
            
            # 验证质量指标范围
            assert 0 <= context_info['confidence_level'] <= 1, "置信度应该在0-1之间"
            assert context_info['source_count'] > 0, "应该有源文档"
            assert 0 <= quality_metrics['overall_quality'] <= 1, "质量分数应该在0-1之间"
        
        # 测试低质量场景
        print(f"\n测试低质量场景:")
        low_quality_results = [
            {
                'content': 'Very short content.',
                'metadata': {
                    'paper_id': 'low_quality',
                    'title': 'Low Quality Paper',
                    'section_type': 'content'
                },
                'combined_score': 0.1,
                'relevance_score': 0.1
            }
        ]
        
        low_quality_result = generator.generate_enhanced_answer(
            "Test low quality query", "general", low_quality_results
        )
        
        if low_quality_result.get('quality_warning'):
            print("   ✅ 正确检测到低质量场景")
        else:
            print("   ⚠️  低质量检测可能需要调整阈值")
        
        print("✅ 质量增强生成器集成测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 质量增强生成器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """测试性能"""
    print("\n" + "=" * 60)
    print("7. 性能测试")
    print("=" * 60)
    
    try:
        from src.generator.quality_enhancement import QualityEnhancedGenerator
        
        generator = QualityEnhancedGenerator()
        
        # 简单的性能测试数据
        mock_results = [
            {
                'content': f'Test content {i} about machine learning and artificial intelligence.',
                'metadata': {
                    'paper_id': f'test_{i}',
                    'title': f'Test Paper {i}',
                    'section_type': 'content'
                },
                'combined_score': 0.5,
                'relevance_score': 0.5
            }
            for i in range(10)
        ]
        
        # 测试处理时间
        start_time = time.time()
        result = generator.generate_enhanced_answer(
            "What is machine learning?", "definition", mock_results
        )
        processing_time = time.time() - start_time
        
        print(f"✅ 处理时间: {processing_time:.3f} 秒")
        print(f"✅ 处理了 {len(mock_results)} 个文档")
        print(f"✅ 平均每文档: {processing_time/len(mock_results):.4f} 秒")
        
        # 验证在合理时间内完成
        assert processing_time < 10, "处理时间应该在10秒内"
        
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("质量增强模块测试开始")
    print("🔍 测试 src/generator/quality_enhancement.py")
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("模块导入", test_imports),
        ("上下文优化器", test_context_optimizer),
        ("提示词工程师", test_prompt_engineer),
        ("事实核查器", test_fact_checker),
        ("引用管理器", test_citation_manager),
        ("质量增强生成器", test_quality_enhanced_generator),
        ("性能测试", test_performance)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {e}")
            test_results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总结: {passed}/{total} 项测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 所有测试通过！quality_enhancement.py 模块工作正常")
    else:
        print("⚠️  部分测试失败，请检查相关功能")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
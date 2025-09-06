#!/usr/bin/env python3
"""
主RAG系统 - 基于工作版本添加混合检索功能
整合了基础检索、混合检索、查询理解等高级功能
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import re

# 添加src到路径
sys.path.append('src')

class MainRAGSystem:
    """主RAG系统 - 整合所有功能的统一接口"""
    
    def __init__(self):
        # 基础组件初始化
        self.vector_store = None
        self.hybrid_retriever = None
        self.quality_generator = None
        self.papers_loaded = False
        
        # 功能开关 - 可以渐进式启用
        self.features = {
            'basic_retrieval': True,      # 基础向量检索
            'hybrid_retrieval': True,     # 混合检索 (BM25 + 向量)
            'query_expansion': True,      # 查询扩展
            'reranking': True,           # 重排序
            'diversity': True,           # 多样性优化
            'intent_detection': True,    # 意图识别
            'quality_enhancement': True,  # 生成质量增强
            'fact_checking': True,       # 事实核查
            'citation_management': True   # 引用管理
        }
        
        # 系统统计
        self.stats = {
            'papers_processed': 0,
            'chunks_generated': 0,
            'queries_handled': 0,
            'system_ready': False
        }
        
        print("主RAG系统初始化完成")
        self._display_feature_status()
    
    def _display_feature_status(self):
        """显示功能状态"""
        print("\n功能状态:")
        for feature, enabled in self.features.items():
            status = "启用" if enabled else "禁用"
            print(f"  {feature}: {status}")
    
    def setup_system(self):
        """设置完整系统"""
        print("\n开始设置主RAG系统...")
        
        # 1. 初始化基础组件
        if not self._initialize_components():
            return False
        
        # 2. 处理论文数据
        if not self._process_papers():
            return False
        
        # 3. 构建检索系统
        if not self._build_retrieval_system():
            return False
        
        # 4. 系统就绪
        self.stats['system_ready'] = True
        print(f"\n系统设置完成！")
        print(f"处理论文: {self.stats['papers_processed']} 篇")
        print(f"文档块: {self.stats['chunks_generated']} 个")
        print(f"混合检索: {'启用' if self.hybrid_retriever else '禁用'}")
        print(f"质量增强: {'启用' if self.quality_generator else '禁用'}")
        
        return True
    
    def _initialize_components(self):
        """初始化基础组件"""
        try:
            # 导入向量存储
            from src.retriever.vector_store import VectorStore
            self.vector_store = VectorStore()
            print("向量存储初始化成功")
            
            # 尝试导入混合检索器
            if self.features['hybrid_retrieval']:
                try:
                    from src.retriever.advanced_retrieval import HybridRetriever
                    print("混合检索模块导入成功")
                except ImportError as e:
                    print(f"混合检索模块导入失败: {e}")
                    print("将使用基础检索模式")
                    self.features['hybrid_retrieval'] = False
            
            # 尝试导入生成质量增强器
            if self.features['quality_enhancement']:
                try:
                    from src.generator.quality_enhancement import QualityEnhancedGenerator
                    self.quality_generator = QualityEnhancedGenerator()
                    print("生成质量增强模块导入成功")
                except ImportError as e:
                    print(f"生成质量增强模块导入失败: {e}")
                    print("将使用基础生成模式")
                    self.features['quality_enhancement'] = False
                    self.features['fact_checking'] = False
                    self.features['citation_management'] = False
            
            return True
            
        except Exception as e:
            print(f"组件初始化失败: {e}")
            return False
    
    def _process_papers(self):
        """处理论文数据"""
        print("\n处理论文数据...")
        
        # 检查论文数据
        papers_file = "data/papers_info.json"
        if not Path(papers_file).exists():
            print(f"论文数据文件不存在: {papers_file}")
            print("请先运行 'python simple_demo.py' 收集论文")
            return False
        
        # 加载论文数据
        with open(papers_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        print(f"找到 {len(papers)} 篇论文")
        
        # 处理每篇论文
        processed_papers = []
        all_chunks = []
        
        for i, paper in enumerate(papers, 1):
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            
            print(f"处理论文 {i}/{len(papers)}: {title[:40]}...")
            
            # 创建文档内容
            content = f"Title: {title}\n\nAbstract: {abstract}"
            
            if len(content.strip()) < 20:
                print(f"  内容太短，跳过")
                continue
            
            # 创建文档块
            chunk = self._create_document_chunk(content, paper['id'], title)
            
            # 保存处理结果
            paper_copy = paper.copy()
            paper_copy['processed_chunks'] = [chunk]
            paper_copy['chunk_count'] = 1
            processed_papers.append(paper_copy)
            all_chunks.append(chunk)
            
            print(f"  生成 1 个文档块")
        
        # 保存处理结果
        output_file = "data/main_system_papers.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_papers, f, ensure_ascii=False, indent=2)
        
        print(f"处理结果保存到: {output_file}")
        
        # 更新统计信息
        self.stats['papers_processed'] = len(processed_papers)
        self.stats['chunks_generated'] = len(all_chunks)
        self.processed_chunks = all_chunks
        
        return len(all_chunks) > 0
    
    def _create_document_chunk(self, content: str, paper_id: str, title: str) -> Dict:
        """创建文档块"""
        # 基础元数据
        metadata = {
            'chunk_id': f"{paper_id}_chunk_0",
            'paper_id': paper_id,
            'title': title,
            'section_type': 'abstract',
            'word_count': len(content.split()),
            'char_count': len(content),
            'has_formulas': bool(re.search(r'\$.*?\$|\\[a-zA-Z]+', content)),
            'has_code': bool(re.search(r'def |class |import |function', content, re.IGNORECASE)),
            'has_citations': bool(re.search(r'\[[0-9,\-\s]+\]|\([A-Za-z]+,?\s*[0-9]{4}\)', content)),
            'has_numbers': bool(re.search(r'\b\d+\.?\d*\b', content))
        }
        
        return {
            'text': content,
            'metadata': metadata,
            'paper_id': paper_id,
            'chunk_index': 0
        }
    
    def _build_retrieval_system(self):
        """构建检索系统"""
        print("\n构建检索系统...")
        
        if not hasattr(self, 'processed_chunks') or not self.processed_chunks:
            print("没有可用的文档块")
            return False
        
        # 1. 构建向量数据库
        print("构建向量数据库...")
        documents = [chunk['text'] for chunk in self.processed_chunks]
        metadatas = [chunk['metadata'] for chunk in self.processed_chunks]
        
        try:
            self.vector_store.add_papers_with_metadata(documents, metadatas)
            print("向量数据库构建成功")
        except Exception as e:
            print(f"向量数据库构建失败: {e}")
            return False
        
        # 2. 初始化混合检索器
        if self.features['hybrid_retrieval']:
            try:
                from src.retriever.advanced_retrieval import HybridRetriever
                print("初始化混合检索器...")
                self.hybrid_retriever = HybridRetriever(self.vector_store)
                self.hybrid_retriever.fit(documents)
                print("混合检索器就绪")
            except Exception as e:
                print(f"混合检索器初始化失败: {e}")
                print("将使用基础检索模式")
                self.features['hybrid_retrieval'] = False
        
        self.papers_loaded = True
        return True
    
    def query(self, question: str, mode: str = "auto") -> Dict:
        """统一查询接口"""
        if not self.papers_loaded:
            return {"error": "系统未就绪，请先运行 setup_system()"}
        
        self.stats['queries_handled'] += 1
        start_time = time.time()
        
        print(f"\n查询: {question}")
        
        # 根据模式选择检索方法
        if mode == "basic" or not self.features['hybrid_retrieval']:
            result = self._basic_query(question)
        elif mode == "enhanced" and self.quality_generator:
            result = self._enhanced_query(question)
        elif mode == "hybrid" or (mode == "auto" and self.hybrid_retriever):
            result = self._hybrid_query(question)
        elif mode == "auto" and self.quality_generator:
            result = self._enhanced_query(question)
        else:
            result = self._basic_query(question)
        
        # 添加查询统计
        query_time = time.time() - start_time
        result['query_time'] = f"{query_time:.2f}秒"
        result['query_count'] = self.stats['queries_handled']
        
        return result
    
    def _basic_query(self, question: str) -> Dict:
        """基础向量检索"""
        print("执行基础向量检索...")
        
        try:
            results = self.vector_store.search(question, top_k=5)
            
            # 构建回答
            answer_parts = ["基础检索结果:\n"]
            
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas']), 1):
                answer_parts.append(f"结果 {i}:")
                answer_parts.append(f"  来源: {metadata.get('title', 'Unknown')}")
                answer_parts.append(f"  内容: {doc[:200]}...\n")
            
            return {
                "question": question,
                "answer": "\n".join(answer_parts),
                "sources": results['metadatas'],
                "search_mode": "basic",
                "results_count": len(results['documents'])
            }
            
        except Exception as e:
            return {"error": f"基础检索失败: {e}"}
    
    def _hybrid_query(self, question: str) -> Dict:
        """混合检索"""
        print("执行混合检索...")
        
        try:
            # 执行混合检索
            hybrid_results = self.hybrid_retriever.search(
                question,
                top_k=5,
                use_reranking=self.features['reranking'],
                use_diversity=self.features['diversity']
            )
            
            # 构建增强回答
            query_intent = hybrid_results['query_intent']
            results = hybrid_results['results']
            stats = hybrid_results['retrieval_stats']
            
            answer_parts = [f"混合检索结果 (意图: {query_intent.intent_type}):\n"]
            
            if self.features['query_expansion']:
                answer_parts.append(f"扩展查询: {query_intent.expanded_query}\n")
            
            for i, result in enumerate(results, 1):
                answer_parts.append(f"结果 {i}:")
                answer_parts.append(f"  来源: {result.metadata.get('title', 'Unknown')}")
                answer_parts.append(f"  章节: {result.metadata.get('section_type', 'unknown')}")
                answer_parts.append(f"  融合分数: {result.combined_score:.3f}")
                
                if self.features['reranking']:
                    answer_parts.append(f"  重排序分数: {result.rerank_score:.3f}")
                
                # 内容特征
                features = self._format_content_features(result.metadata)
                if features:
                    answer_parts.append(f"  特征: {features}")
                
                answer_parts.append(f"  内容: {result.content[:200]}...\n")
            
            return {
                "question": question,
                "answer": "\n".join(answer_parts),
                "query_intent": query_intent.intent_type,
                "expanded_query": query_intent.expanded_query,
                "sources": [r.metadata for r in results],
                "retrieval_stats": stats,
                "search_mode": "hybrid",
                "results_count": len(results),
                "features_used": [k for k, v in self.features.items() if v]
            }
            
        except Exception as e:
            print(f"混合检索失败，回退到基础检索: {e}")
            return self._basic_query(question)
    
    def _enhanced_query(self, question: str) -> Dict:
        """质量增强检索"""
        print("执行质量增强检索...")
        
        try:
            # 首先获取检索结果
            if self.hybrid_retriever:
                # 使用混合检索获取更好的结果
                hybrid_results = self.hybrid_retriever.search(
                    question,
                    top_k=8,  # 获取更多结果供质量增强器筛选
                    use_reranking=self.features['reranking'],
                    use_diversity=self.features['diversity']
                )
                retrieved_results = []
                query_intent = hybrid_results['query_intent'].intent_type
                
                for result in hybrid_results['results']:
                    retrieved_results.append({
                        'content': result.content,
                        'metadata': result.metadata,
                        'combined_score': result.combined_score,
                        'relevance_score': getattr(result, 'relevance_score', result.combined_score)
                    })
            else:
                # 回退到基础检索
                basic_results = self.vector_store.search(question, top_k=8)
                retrieved_results = []
                query_intent = "general"
                
                for doc, metadata in zip(basic_results['documents'], basic_results['metadatas']):
                    retrieved_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'combined_score': 0.5,
                        'relevance_score': 0.5
                    })
            
            # 使用质量增强生成器
            enhanced_result = self.quality_generator.generate_enhanced_answer(
                query=question,
                query_intent=query_intent,
                retrieved_results=retrieved_results
            )
            
            # 检查质量警告
            if enhanced_result.get('quality_warning', False):
                return {
                    "question": question,
                    "answer": enhanced_result['answer'],
                    "search_mode": "enhanced",
                    "quality_warning": True,
                    "confidence": enhanced_result['confidence']
                }
            
            # 构建完整的增强回答
            answer_parts = ["质量增强检索结果:\n"]
            
            # 添加质量信息
            context_info = enhanced_result['context_info']
            answer_parts.append(f"置信度: {self._get_confidence_description(context_info['confidence_level'])}")
            answer_parts.append(f"参考源: {context_info['source_count']} 篇论文")
            
            # 添加生成的回答
            answer_parts.append(f"\n{enhanced_result['answer']}")
            
            # 添加事实核查结果
            if enhanced_result.get('fact_check') and self.features['fact_checking']:
                fact_checks = enhanced_result['fact_check']
                if fact_checks:
                    answer_parts.append(f"\n事实核查结果:")
                    for i, fc in enumerate(fact_checks[:3], 1):  # 限制显示数量
                        answer_parts.append(f"  {i}. 声明: {fc.claim[:100]}...")
                        answer_parts.append(f"     支持度: {fc.support_level} (置信度: {fc.confidence_score:.2f})")
            
            # 添加引用信息
            if enhanced_result.get('citations') and self.features['citation_management']:
                citations = enhanced_result['citations']
                answer_parts.append(f"\n学术引用:")
                for i, citation in enumerate(citations[:5], 1):
                    citation_text = self.quality_generator.citation_manager.format_citation(citation)
                    answer_parts.append(f"  [{i}] {citation_text}")
            
            # 添加质量指标
            quality_metrics = enhanced_result['quality_metrics']
            answer_parts.append(f"\n质量评估:")
            answer_parts.append(f"  整体质量: {quality_metrics['overall_quality']:.2f}/1.0")
            answer_parts.append(f"  建议: {quality_metrics['recommendation']}")
            
            return {
                "question": question,
                "answer": "\n".join(answer_parts),
                "query_intent": query_intent,
                "sources": [r['metadata'] for r in retrieved_results],
                "search_mode": "enhanced",
                "results_count": len(retrieved_results),
                "quality_metrics": quality_metrics,
                "context_info": context_info,
                "citations": enhanced_result.get('citations', []),
                "fact_check": enhanced_result.get('fact_check', []),
                "features_used": [k for k, v in self.features.items() if v]
            }
            
        except Exception as e:
            print(f"质量增强检索失败，回退到混合检索: {e}")
            if self.hybrid_retriever:
                return self._hybrid_query(question)
            else:
                return self._basic_query(question)
    
    def _get_confidence_description(self, confidence: float) -> str:
        """获取置信度描述"""
        if confidence >= 0.8:
            return f"高 ({confidence:.2f}) - 基于多个权威源"
        elif confidence >= 0.6:
            return f"中等 ({confidence:.2f}) - 基于相关源"
        elif confidence >= 0.4:
            return f"较低 ({confidence:.2f}) - 证据有限"
        else:
            return f"低 ({confidence:.2f}) - 证据不足"
    
    def _format_content_features(self, metadata: Dict) -> str:
        """格式化内容特征"""
        features = []
        if metadata.get('has_formulas'): features.append("公式")
        if metadata.get('has_code'): features.append("代码")
        if metadata.get('has_citations'): features.append("引用")
        if metadata.get('has_numbers'): features.append("数据")
        return ", ".join(features)
    
    def toggle_feature(self, feature_name: str) -> bool:
        """切换功能开关"""
        if feature_name in self.features:
            self.features[feature_name] = not self.features[feature_name]
            print(f"功能 {feature_name}: {'启用' if self.features[feature_name] else '禁用'}")
            return True
        else:
            print(f"未知功能: {feature_name}")
            return False
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return {
            "stats": self.stats,
            "features": self.features,
            "components": {
                "vector_store": self.vector_store is not None,
                "hybrid_retriever": self.hybrid_retriever is not None,
                "quality_generator": self.quality_generator is not None,
                "papers_loaded": self.papers_loaded
            }
        }
    
    def benchmark_retrieval(self, test_queries: List[str] = None) -> Dict:
        """检索系统性能基准测试"""
        if not test_queries:
            test_queries = [
                "transformer attention mechanism",
                "neural network optimization",
                "machine learning algorithms",
                "deep learning applications",
                "artificial intelligence methods"
            ]
        
        print(f"\n执行检索基准测试...")
        results = {
            "basic_retrieval": [],
            "hybrid_retrieval": [],
            "enhanced_retrieval": []
        }
        
        for query in test_queries:
            print(f"测试查询: {query}")
            
            # 基础检索测试
            start_time = time.time()
            basic_result = self._basic_query(query)
            basic_time = time.time() - start_time
            results["basic_retrieval"].append({
                "query": query,
                "time": basic_time,
                "results_count": basic_result.get("results_count", 0)
            })
            
            # 混合检索测试
            if self.hybrid_retriever:
                start_time = time.time()
                hybrid_result = self._hybrid_query(query)
                hybrid_time = time.time() - start_time
                results["hybrid_retrieval"].append({
                    "query": query,
                    "time": hybrid_time,
                    "results_count": hybrid_result.get("results_count", 0),
                    "features_used": hybrid_result.get("features_used", [])
                })
            
            # 质量增强检索测试
            if self.quality_generator:
                start_time = time.time()
                enhanced_result = self._enhanced_query(query)
                enhanced_time = time.time() - start_time
                results["enhanced_retrieval"].append({
                    "query": query,
                    "time": enhanced_time,
                    "results_count": enhanced_result.get("results_count", 0),
                    "features_used": enhanced_result.get("features_used", []),
                    "quality_score": enhanced_result.get("quality_metrics", {}).get("overall_quality", 0),
                    "confidence": enhanced_result.get("context_info", {}).get("confidence_level", 0)
                })
        
        # 计算平均性能
        avg_basic_time = sum(r["time"] for r in results["basic_retrieval"]) / len(results["basic_retrieval"])
        
        summary = {
            "test_queries": len(test_queries),
            "basic_avg_time": f"{avg_basic_time:.3f}秒",
            "detailed_results": results
        }
        
        if results["hybrid_retrieval"]:
            avg_hybrid_time = sum(r["time"] for r in results["hybrid_retrieval"]) / len(results["hybrid_retrieval"])
            summary["hybrid_avg_time"] = f"{avg_hybrid_time:.3f}秒"
            summary["performance_ratio"] = f"{avg_hybrid_time/avg_basic_time:.2f}x"
        
        if results["enhanced_retrieval"]:
            avg_enhanced_time = sum(r["time"] for r in results["enhanced_retrieval"]) / len(results["enhanced_retrieval"])
            avg_quality_score = sum(r["quality_score"] for r in results["enhanced_retrieval"]) / len(results["enhanced_retrieval"])
            avg_confidence = sum(r["confidence"] for r in results["enhanced_retrieval"]) / len(results["enhanced_retrieval"])
            summary["enhanced_avg_time"] = f"{avg_enhanced_time:.3f}秒"
            summary["enhanced_performance_ratio"] = f"{avg_enhanced_time/avg_basic_time:.2f}x"
            summary["avg_quality_score"] = f"{avg_quality_score:.3f}"
            summary["avg_confidence"] = f"{avg_confidence:.3f}"
        
        return summary


def main():
    """主程序"""
    print("主RAG系统启动")
    print("=" * 60)
    
    # 初始化系统
    rag = MainRAGSystem()
    
    # 设置系统
    print("\n正在设置系统...")
    if not rag.setup_system():
        print("系统设置失败，退出")
        return
    
    # 显示系统状态
    status = rag.get_system_status()
    print(f"\n系统状态: {'就绪' if status['stats']['system_ready'] else '未就绪'}")
    
    # 交互式查询
    print("\n系统就绪！可用命令:")
    print("  'quit' - 退出")
    print("  'status' - 查看系统状态") 
    print("  'benchmark' - 运行性能测试")
    print("  'basic <查询>' - 使用基础检索")
    print("  'hybrid <查询>' - 使用混合检索")
    print("  'enhanced <查询>' - 使用质量增强检索")
    print("  'toggle <功能名>' - 切换功能开关")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n请输入命令或问题: ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break
            
            elif user_input.lower() == 'status':
                status = rag.get_system_status()
                print(f"\n系统状态:")
                print(f"  论文: {status['stats']['papers_processed']} 篇")
                print(f"  文档块: {status['stats']['chunks_generated']} 个") 
                print(f"  查询次数: {status['stats']['queries_handled']}")
                print(f"  功能状态: {status['features']}")
                
            elif user_input.lower() == 'benchmark':
                print("运行性能基准测试...")
                benchmark_results = rag.benchmark_retrieval()
                print(f"\n基准测试结果:")
                print(f"  测试查询数: {benchmark_results['test_queries']}")
                print(f"  基础检索平均时间: {benchmark_results['basic_avg_time']}")
                if 'hybrid_avg_time' in benchmark_results:
                    print(f"  混合检索平均时间: {benchmark_results['hybrid_avg_time']}")
                    print(f"  性能比率: {benchmark_results['performance_ratio']}")
                if 'enhanced_avg_time' in benchmark_results:
                    print(f"  质量增强检索平均时间: {benchmark_results['enhanced_avg_time']}")
                    print(f"  增强性能比率: {benchmark_results['enhanced_performance_ratio']}")
                    print(f"  平均质量分数: {benchmark_results['avg_quality_score']}")
                    print(f"  平均置信度: {benchmark_results['avg_confidence']}")
                
            elif user_input.startswith('toggle '):
                feature_name = user_input[7:].strip()
                rag.toggle_feature(feature_name)
                
            elif user_input.startswith('basic '):
                question = user_input[6:].strip()
                if question:
                    result = rag.query(question, mode="basic")
                    if "error" not in result:
                        print(f"\n{result['answer']}")
                        print(f"查询耗时: {result['query_time']}")
                    else:
                        print(f"错误: {result['error']}")
                        
            elif user_input.startswith('hybrid '):
                question = user_input[7:].strip()
                if question:
                    result = rag.query(question, mode="hybrid")
                    if "error" not in result:
                        print(f"\n{result['answer']}")
                        print(f"查询耗时: {result['query_time']}")
                        if 'retrieval_stats' in result:
                            print(f"检索统计: {result['retrieval_stats']}")
                    else:
                        print(f"错误: {result['error']}")
                        
            elif user_input.startswith('enhanced '):
                question = user_input[9:].strip()
                if question:
                    result = rag.query(question, mode="enhanced")
                    if "error" not in result:
                        print(f"\n{result['answer']}")
                        print(f"查询耗时: {result['query_time']}")
                        if 'quality_metrics' in result:
                            print(f"质量评分: {result['quality_metrics']['overall_quality']:.2f}")
                        if 'context_info' in result:
                            print(f"置信度: {result['context_info']['confidence_level']:.2f}")
                    else:
                        print(f"错误: {result['error']}")
            
            else:
                # 默认自动检索
                result = rag.query(user_input, mode="auto")
                if "error" not in result:
                    print(f"\n{result['answer']}")
                    print(f"检索模式: {result.get('search_mode', 'unknown')}")
                    print(f"查询耗时: {result['query_time']}")
                else:
                    print(f"错误: {result['error']}")
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    main()

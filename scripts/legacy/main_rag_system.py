#!/usr/bin/env python3
"""
主RAG系统 - 基于trace_demo.py的成功模式重写
专注于生成高质量的中文学术回答
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# 添加src到路径
sys.path.append('src')

class MainRAGSystem:
    """主RAG系统 - 简化版，专注于高质量回答生成"""
    
    def __init__(self):
        """初始化系统"""
        self.vector_store = None
        self.hybrid_retriever = None
        self.llm_manager = None
        self.papers_loaded = False
        
        print("🚀 主RAG系统初始化...")
        
    def setup_system(self):
        """设置完整系统"""
        print("\n📚 开始设置RAG系统...")
        
        try:
            # 1. 初始化向量存储
            if not self._initialize_vector_store():
                return False
            
            # 2. 处理论文数据
            if not self._load_papers():
                return False
            
            # 3. 初始化混合检索器
            if not self._initialize_hybrid_retriever():
                return False
            
            # 4. 初始化LLM管理器
            if not self._initialize_llm_manager():
                return False
            
            print(f"\n✅ 系统设置完成！")
            print(f"📊 处理论文: {len(self.processed_papers)} 篇")
            print(f"🔍 混合检索: 已启用")
            print(f"🤖 LLM模型: llama3.1:8b")
            
            self.papers_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ 系统设置失败: {e}")
            return False
    
    def _initialize_vector_store(self):
        """初始化增强版向量存储 - 确保嵌入模型一致性"""
        try:
            from src.retriever.enhanced_vector_store import EnhancedVectorStore
            from src.config.embedding_config import SystemEmbeddingConfig
            
            # 使用系统统一嵌入配置 - 确保向量空间一致性
            embedding_config = SystemEmbeddingConfig.get_default_config()
            
            print(f"🔧 使用系统统一嵌入配置:")
            print(f"   - 模型: {embedding_config.model_name}")
            print(f"   - 维度: {SystemEmbeddingConfig.DEFAULT_DIMENSION}")
            print(f"   - 归一化: {embedding_config.normalize_embeddings}")
            print(f"   - 缓存: {embedding_config.cache_enabled}")
            
            self.vector_store = EnhancedVectorStore(
                persist_directory="vector_db",
                embedding_config=embedding_config
            )
            print("✅ 增强版向量存储初始化成功")
            return True
        except Exception as e:
            print(f"❌ 增强版向量存储初始化失败: {e}")
            print("⚠️ 回退到基础向量存储（使用相同嵌入模型确保一致性）...")
            try:
                from src.retriever.vector_store import VectorStore
                from src.config.embedding_config import SystemEmbeddingConfig
                
                # 即使在回退情况下，也必须使用相同的嵌入模型
                # 检查基础向量存储是否使用相同模型
                if hasattr(VectorStore, '__init__'):
                    # 创建基础向量存储，但确保使用相同嵌入模型
                    self.vector_store = VectorStore()
                    
                    # 验证基础向量存储的嵌入模型是否一致
                    expected_model = SystemEmbeddingConfig.DEFAULT_MODEL_NAME
                    if hasattr(self.vector_store, 'encoder'):
                        actual_model = getattr(self.vector_store.encoder, 'model_name', 'unknown')
                        if 'all-MiniLM-L6-v2' in str(actual_model):
                            print("❌ 基础向量存储使用了不同的嵌入模型!")
                            print(f"   期望: {expected_model}")
                            print(f"   实际: {actual_model}")
                            print("🚨 向量空间不一致，无法安全回退!")
                            return False
                    
                    print("✅ 基础向量存储初始化成功（嵌入模型一致）")
                    return True
                else:
                    return False
            except Exception as e2:
                print(f"❌ 基础向量存储也失败: {e2}")
                return False
    
    def _load_papers(self):
        """加载论文数据并进行智能切分"""
        try:
            papers_file = "data/papers_info.json"
            if not Path(papers_file).exists():
                print(f"❌ 论文数据文件不存在: {papers_file}")
                return False
            
            with open(papers_file, 'r', encoding='utf-8') as f:
                self.papers_data = json.load(f)
            
            print(f"📄 加载论文数据: {len(self.papers_data)} 篇")
            
            # 初始化文档切分器
            from src.processor.document_chunker import DocumentChunker, ChunkingConfig
            
            chunking_config = ChunkingConfig(
                strategy="hybrid",           # 使用混合策略
                chunk_size=600,             # 块大小600字符
                chunk_overlap=100,          # 重叠100字符
                min_chunk_size=150,         # 最小块大小
                max_chunk_size=1200,        # 最大块大小
                preserve_paragraphs=True,   # 保持段落完整
                preserve_sentences=True,    # 保持句子完整
                section_aware=True          # 感知章节结构
            )
            
            self.document_chunker = DocumentChunker(chunking_config)
            print(f"🔪 文档切分配置: {chunking_config.strategy}策略, 块大小{chunking_config.chunk_size}, 重叠{chunking_config.chunk_overlap}")
            
            # 处理论文并进行智能切分
            self.processed_papers = []
            all_chunks = []
            documents = []
            metadatas = []
            
            total_chunks = 0
            
            for i, paper in enumerate(self.papers_data):
                title = paper.get('title', '')
                abstract = paper.get('abstract', '')
                
                if len(title) < 10 or len(abstract) < 50:
                    continue
                
                # 创建完整文档内容
                content = f"Title: {title}\n\nAbstract: {abstract}"
                
                # 基础元数据
                base_metadata = {
                    'paper_id': paper['id'],
                    'title': title,
                    'authors': paper.get('authors', []),
                    'published': paper.get('published', ''),
                    'pdf_url': paper.get('pdf_url', ''),
                    'source_paper': paper
                }
                
                # 使用文档切分器进行智能切分
                chunks = self.document_chunker.chunk_document(content, paper['id'], base_metadata)
                
                if chunks:
                    # 收集所有chunks
                    all_chunks.extend(chunks)
                    total_chunks += len(chunks)
                    
                    # 转换为向量存储格式
                    for chunk in chunks:
                        documents.append(chunk.text)
                        metadatas.append(chunk.metadata)
                    
                    # 保存处理后的论文信息
                    paper_copy = paper.copy()
                    paper_copy['chunks'] = [
                        {
                            'chunk_id': chunk.chunk_id,
                            'text': chunk.text,
                            'section_type': chunk.section_type,
                            'word_count': chunk.word_count,
                            'char_count': chunk.char_count
                        }
                        for chunk in chunks
                    ]
                    paper_copy['chunk_count'] = len(chunks)
                    self.processed_papers.append(paper_copy)
                
                if (i + 1) % 10 == 0:
                    print(f"  已处理 {i + 1}/{len(self.papers_data)} 篇论文...")
            
            # 获取切分统计信息
            chunking_stats = self.document_chunker.get_chunking_stats(all_chunks)
            
            print(f"\n📊 文档切分统计:")
            print(f"  总文档块数: {chunking_stats['total_chunks']}")
            print(f"  平均块大小: {chunking_stats['avg_chunk_size']:.1f} 字符")
            print(f"  平均词数: {chunking_stats['avg_word_count']:.1f}")
            print(f"  块大小范围: {chunking_stats['min_chunk_size']} - {chunking_stats['max_chunk_size']} 字符")
            print(f"  识别章节类型: {', '.join(chunking_stats['section_types'])}")
            
            # 构建向量数据库
            print("\n🔢 构建向量数据库...")
            self.vector_store.add_papers_with_metadata(documents, metadatas)
            self.documents = documents
            self.all_chunks = all_chunks
            
            print(f"✅ 处理完成: {len(self.processed_papers)} 篇论文, {total_chunks} 个智能文档块")
            return True
            
        except Exception as e:
            print(f"❌ 论文数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _initialize_hybrid_retriever(self):
        """初始化混合检索器"""
        try:
            from src.retriever.advanced_retrieval import HybridRetriever
            self.hybrid_retriever = HybridRetriever(self.vector_store)
            self.hybrid_retriever.fit(self.documents)
            print("✅ 混合检索器初始化成功")
            return True
        except Exception as e:
            print(f"❌ 混合检索器初始化失败: {e}")
            return False
    
    def _initialize_llm_manager(self):
        """初始化LLM管理器"""
        try:
            from src.generator.llm_client import get_llm_manager
            self.llm_manager = get_llm_manager("llama3.1:8b")
            print("✅ LLM管理器初始化成功")
            return True
        except Exception as e:
            print(f"❌ LLM管理器初始化失败: {e}")
            return False
    
    def query(self, question: str) -> Dict:
        """处理用户查询并生成高质量回答"""
        if not self.papers_loaded:
            return {"error": "系统未就绪，请先运行setup_system()"}
        
        start_time = time.time()
        print(f"\n🔍 处理查询: {question}")
        
        try:
            # 1. 执行增强向量检索
            print("  🎯 执行增强向量检索 (带相似度阈值过滤)...")
            enhanced_results = self.vector_store.advanced_search(
                query=question,
                top_k=8,
                similarity_threshold=0.3,
                enable_adaptive_k=True,
                enable_diversity=False
            )
            
            # 2. 执行混合检索作为补充
            print("  📖 执行混合检索...")
            hybrid_results = self.hybrid_retriever.search(
                question, 
                top_k=5, 
                use_reranking=True, 
                use_diversity=True
            )
            
            # 3. 合并和优化检索结果
            print(f"  📊 增强检索结果: {len(enhanced_results)} 个")
            
            # 获取混合检索结果
            final_results = hybrid_results.get('final_results', hybrid_results.get('results', []))
            print(f"  📊 混合检索结果: {len(final_results)} 个")
            
            # 优先使用增强向量检索结果，如果结果不足则补充混合检索结果
            if enhanced_results:
                # 将增强检索结果转换为混合检索格式
                enhanced_docs = []
                for result in enhanced_results:
                    enhanced_docs.append({
                        'content': result.document,
                        'metadata': result.metadata,
                        'similarity_score': result.similarity_score,
                        'source': 'enhanced_vector'
                    })
                
                # 如果增强检索结果充足，直接使用；否则补充混合检索结果
                if len(enhanced_docs) >= 3:
                    final_results = enhanced_docs[:5]  # 取前5个最佳结果
                else:
                    final_results = enhanced_docs + final_results[:max(0, 5-len(enhanced_docs))]
            
            if not final_results:
                return {
                    "question": question,
                    "answer": "抱歉，未找到相关的学术资料来回答您的问题。建议降低相似度阈值重试。",
                    "query_time": f"{time.time() - start_time:.2f}秒",
                    "retrieval_info": {
                        "enhanced_results": len(enhanced_results),
                        "hybrid_results": len(hybrid_results.get('final_results', []))
                    }
                }
            
            print(f"  ✅ 检索到 {len(final_results)} 个相关文档")
            
            # 3. 构建增强Prompt (使用先进的提示工程)
            print("  🛠️ 构建智能提示词...")
            from src.generator.prompt_engineering import PromptBuilder
            
            prompt_builder = PromptBuilder()
            
            # 直接传递原始检索结果给提示工程模块
            # 提示工程模块会内部处理和转换数据格式
            prompt_result = prompt_builder.build_prompt(
                query=question,
                retrieved_results=final_results,
                max_context_length=4000  # 控制提示词长度
            )
            
            # 4. 调用LLM生成回答
            print("  🤖 生成智能回答...")
            
            response = self.llm_manager.generate_answer(
                prompt=prompt_result["prompt"],
                query_intent=prompt_result["query_type"],
                max_tokens=512,
                temperature=0.7
            )
            
            query_time = time.time() - start_time
            
            if response.success and response.text.strip():
                return {
                    "question": question,
                    "answer": response.text.strip(),
                    "sources": [r.get('metadata', {}) if isinstance(r, dict) else r.metadata for r in final_results],
                    "query_time": f"{query_time:.2f}秒",
                    "model": response.model,
                    "results_count": len(final_results)
                }
            else:
                return {
                    "question": question,
                    "error": f"LLM生成失败: {response.error_message}",
                    "query_time": f"{query_time:.2f}秒"
                }
                
        except Exception as e:
            return {
                "question": question,
                "error": f"查询处理失败: {str(e)}",
                "query_time": f"{time.time() - start_time:.2f}秒"
            }
    
    def get_system_status(self):
        """获取系统状态"""
        status = {
            "papers_loaded": self.papers_loaded,
            "papers_count": len(self.processed_papers) if hasattr(self, 'processed_papers') else 0,
            "vector_store_ready": self.vector_store is not None,
            "hybrid_retriever_ready": self.hybrid_retriever is not None,
            "llm_manager_ready": self.llm_manager is not None
        }
        
        # 添加文档切分统计信息
        if hasattr(self, 'all_chunks') and self.all_chunks:
            chunking_stats = self.document_chunker.get_chunking_stats(self.all_chunks)
            status.update({
                "total_chunks": chunking_stats['total_chunks'],
                "avg_chunk_size": f"{chunking_stats['avg_chunk_size']:.1f} 字符",
                "chunks_per_paper": f"{chunking_stats['total_chunks'] / len(self.processed_papers):.1f}" if self.processed_papers else "0",
                "chunking_strategy": self.document_chunker.config.strategy,
                "section_types": chunking_stats['section_types']
            })
        
        # 添加嵌入模型信息
        if hasattr(self.vector_store, 'get_collection_stats'):
            try:
                vector_stats = self.vector_store.get_collection_stats()
                status.update({
                    "embedding_model": vector_stats.get("embedding_model", "unknown"),
                    "embedding_dimension": vector_stats.get("embedding_dimension", "unknown"),
                    "embedding_device": vector_stats.get("device", "unknown"),
                    "embedding_cache": "启用" if vector_stats.get("cache_enabled", False) else "禁用"
                })
            except:
                pass
        
        return status


def main():
    """主程序入口"""
    print("🎯 学术RAG问答系统")
    print("=" * 50)
    
    # 初始化系统
    rag = MainRAGSystem()
    
    # 设置系统
    if not rag.setup_system():
        print("❌ 系统设置失败，退出")
        return
    
    # 显示使用说明
    print(f"\n💡 系统就绪！使用说明:")
    print(f"  • 直接输入问题进行查询")
    print(f"  • 输入 'status' 查看系统状态")
    print(f"  • 输入 'quit' 或 'exit' 退出系统")
    print(f"  • 输入 'help' 查看帮助")
    print("=" * 50)
    
    # 交互式查询循环
    while True:
        try:
            user_input = input("\n❓ 请输入您的问题: ").strip()
            
            if not user_input:
                continue
            
            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 感谢使用，再见！")
                break
            
            elif user_input.lower() == 'status':
                status = rag.get_system_status()
                print(f"\n📊 系统状态:")
                print(f"  论文数据: {'✅' if status['papers_loaded'] else '❌'} ({status['papers_count']} 篇)")
                print(f"  向量存储: {'✅' if status['vector_store_ready'] else '❌'}")
                print(f"  混合检索: {'✅' if status['hybrid_retriever_ready'] else '❌'}")
                print(f"  LLM模型: {'✅' if status['llm_manager_ready'] else '❌'}")
                
                # 显示文档切分信息
                if 'total_chunks' in status:
                    print(f"\n🔪 文档切分详情:")
                    print(f"  切分策略: {status['chunking_strategy']}")
                    print(f"  总文档块: {status['total_chunks']} 个")
                    print(f"  平均块大小: {status['avg_chunk_size']}")
                    print(f"  每篇论文块数: {status['chunks_per_paper']} 个")
                    print(f"  识别章节类型: {', '.join(status['section_types'])}")
                
                # 显示嵌入模型信息
                if 'embedding_model' in status:
                    print(f"\n🔢 嵌入模型详情:")
                    print(f"  模型名称: {status['embedding_model']}")
                    print(f"  向量维度: {status['embedding_dimension']}")
                    print(f"  计算设备: {status['embedding_device']}")
                    print(f"  缓存状态: {status['embedding_cache']}")
                continue
            
            elif user_input.lower() == 'help':
                print(f"\n📖 帮助信息:")
                print(f"  本系统是基于学术论文的智能问答系统")
                print(f"  • 支持中英文问题")
                print(f"  • 基于混合检索技术")
                print(f"  • 使用本地LLM生成回答")
                print(f"  • 自动引用相关学术资源")
                print(f"\n💡 示例问题:")
                print(f"  - 什么是transformer架构？")
                print(f"  - 解释注意力机制的原理")
                print(f"  - 比较CNN和RNN的优缺点")
                continue
            
            # 处理查询
            print("⏳ 正在处理您的问题...")
            result = rag.query(user_input)
            
            if "error" not in result:
                print(f"\n📝 智能回答:")
                print("-" * 50)
                print(result['answer'])
                print("-" * 50)
                print(f"⏱️  查询耗时: {result['query_time']}")
                print(f"📚 参考源: {result['results_count']} 篇论文")
                print(f"🤖 模型: {result.get('model', 'Unknown')}")
            else:
                print(f"\n❌ 查询失败: {result['error']}")
                if 'query_time' in result:
                    print(f"⏱️  查询耗时: {result['query_time']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 感谢使用，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
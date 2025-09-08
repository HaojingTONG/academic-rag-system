# src/retriever/enhanced_vector_store.py
"""
增强版向量存储 - 使用高级嵌入功能
"""

import chromadb
import numpy as np
from typing import List, Dict, Optional, Union
import sys
from pathlib import Path

# 添加embedding模块路径
sys.path.append(str(Path(__file__).parent.parent))
from embedding.advanced_embedding import EmbeddingManager, EmbeddingConfig

class EnhancedVectorStore:
    """增强版向量存储"""
    
    def __init__(self, 
                 persist_directory="vector_db",
                 embedding_model="all-mpnet-base-v2",  # 默认使用高质量模型
                 embedding_config: EmbeddingConfig = None):
        """
        初始化增强版向量存储
        
        Args:
            persist_directory: 持久化目录
            embedding_model: 嵌入模型名称
            embedding_config: 自定义嵌入配置
        """
        
        print(f"🚀 初始化增强版向量存储...")
        
        # 初始化ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="enhanced_ai_papers_v2",  # 新版本collection
            metadata={"hnsw:space": "cosine"}
        )
        
        # 初始化嵌入管理器
        if embedding_config is None:
            embedding_config = EmbeddingConfig(
                model_name=embedding_model,
                model_type="sentence_transformers",
                batch_size=16,
                normalize_embeddings=True,
                cache_enabled=True,
                cache_dir="data/embedding_cache"
            )
        
        self.embedding_config = embedding_config
        self.embedding_manager = EmbeddingManager(embedding_config)
        self.embedding_manager.initialize()
        
        # 获取模型信息
        self.model_info = self.embedding_manager.get_model_info()
        
        print(f"✅ 向量存储初始化完成")
        print(f"   - 嵌入模型: {self.model_info.get('model_name', 'unknown')}")
        print(f"   - 向量维度: {self.model_info.get('dimension', 'unknown')}")
        print(f"   - 计算设备: {self.model_info.get('device', 'unknown')}")
        
    def add_papers_with_metadata(self, documents: List[str], metadatas: List[Dict]):
        """添加带有增强元数据的文档块"""
        if not documents or not metadatas:
            print("⚠️ 文档或元数据为空")
            return
        
        print(f"📝 添加 {len(documents)} 个文档块到增强向量数据库...")
        
        # 生成唯一ID
        ids = []
        for i, metadata in enumerate(metadatas):
            chunk_id = metadata.get('chunk_id', f'chunk_{i}')
            paper_id = metadata.get('paper_id', 'unknown')
            ids.append(f"enhanced_{paper_id}_{chunk_id}")
        
        # 生成嵌入向量
        print("🔢 生成高级嵌入向量...")
        start_time = __import__('time').time()
        
        embeddings = self.embedding_manager.encode(
            documents, 
            show_progress=True, 
            use_cache=True
        )
        
        encoding_time = __import__('time').time() - start_time
        print(f"⏱️ 嵌入生成耗时: {encoding_time:.2f}秒")
        
        # 批量添加到数据库
        batch_size = 50  # 适中的批次大小
        total_added = 0
        
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            batch_embeddings = embeddings[i:end_idx]
            batch_documents = documents[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            # 确保嵌入数据类型正确
            if isinstance(batch_embeddings, np.ndarray):
                batch_embeddings = batch_embeddings.tolist()
            
            try:
                self.collection.add(
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                
                total_added += len(batch_documents)
                print(f"✅ 已添加 {total_added}/{len(documents)} 个文档块")
                
            except Exception as e:
                print(f"❌ 批次 {i//batch_size + 1} 添加失败: {e}")
                continue
        
        # 显示嵌入统计
        stats = self.embedding_manager.get_stats()
        print(f"\n📊 嵌入统计:")
        print(f"   - 总编码数量: {stats['total_encoded']}")
        print(f"   - 平均编码时间: {stats['avg_encoding_time']:.3f}秒/文本")
        print(f"   - 缓存命中: {stats.get('cache_hits', 0)}")
        
        print("✅ 增强文档块添加完成")
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Dict = None) -> Dict:
        """搜索相关文档（使用高级嵌入）"""
        
        # 使用高级嵌入模型生成查询向量
        query_embedding = self.embedding_manager.encode(query, show_progress=False)
        
        # 确保查询嵌入格式正确
        if isinstance(query_embedding, np.ndarray):
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.tolist()
        
        # 构建搜索参数
        search_params = {
            "query_embeddings": query_embedding,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        # 添加元数据过滤
        if filter_metadata:
            search_params["where"] = filter_metadata
        
        # 执行搜索
        try:
            results = self.collection.query(**search_params)
            
            # 格式化返回结果
            return {
                "documents": results.get("documents", [[]])[0],
                "metadatas": results.get("metadatas", [[]])[0], 
                "distances": results.get("distances", [[]])[0]
            }
            
        except Exception as e:
            print(f"❌ 搜索失败: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": []
            }
    
    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        try:
            collection_info = self.collection.count()
            
            return {
                "total_documents": collection_info,
                "embedding_model": self.model_info.get("model_name", "unknown"),
                "embedding_dimension": self.model_info.get("dimension", "unknown"),
                "device": self.model_info.get("device", "unknown"),
                "cache_enabled": self.embedding_config.cache_enabled,
                "embedding_stats": self.embedding_manager.get_stats()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def similarity_search_with_scores(self, query: str, top_k: int = 5) -> List[tuple]:
        """相似性搜索并返回分数"""
        results = self.search(query, top_k)
        
        search_results = []
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])
        
        for doc, metadata, distance in zip(documents, metadatas, distances):
            # 将距离转换为相似度分数 (1 - cosine_distance)
            similarity_score = 1.0 - distance
            search_results.append((doc, metadata, similarity_score))
        
        return search_results
    
    def advanced_search(self, 
                       query: str,
                       top_k: int = 5,
                       similarity_threshold: float = 0.3,
                       filter_metadata: Dict = None,
                       enable_adaptive_k: bool = True,
                       enable_diversity: bool = False):
        """
        高级向量检索 - 支持相似度阈值过滤和智能Top-K选择
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值
            filter_metadata: 元数据过滤
            enable_adaptive_k: 启用自适应K值调整
            enable_diversity: 启用结果多样性
            
        Returns:
            增强检索结果
        """
        from src.retriever.enhanced_vector_retrieval import EnhancedVectorRetriever, RetrievalConfig
        
        # 创建检索配置
        config = RetrievalConfig(
            default_top_k=top_k,
            similarity_threshold=similarity_threshold,
            enable_adaptive_k=enable_adaptive_k,
            enable_diversity=enable_diversity
        )
        
        # 创建增强检索器
        retriever = EnhancedVectorRetriever(self, config)
        
        # 执行检索
        return retriever.search(
            query=query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter_metadata=filter_metadata,
            enable_adaptive_k=enable_adaptive_k,
            enable_diversity=enable_diversity
        )
    
    def delete_collection(self):
        """删除整个集合"""
        try:
            self.client.delete_collection(name="enhanced_ai_papers_v2")
            print("✅ 集合删除成功")
        except Exception as e:
            print(f"❌ 集合删除失败: {e}")
    
    def rebuild_embeddings(self, new_model_name: str):
        """使用新模型重建嵌入"""
        print(f"🔄 准备使用新模型重建嵌入: {new_model_name}")
        
        # 获取现有文档
        try:
            all_results = self.collection.get()
            documents = all_results.get("documents", [])
            metadatas = all_results.get("metadatas", [])
            
            if not documents:
                print("⚠️ 没有找到现有文档")
                return
            
            print(f"📄 找到 {len(documents)} 个现有文档")
            
            # 删除现有集合
            self.delete_collection()
            
            # 重新初始化使用新模型
            new_config = EmbeddingConfig(
                model_name=new_model_name,
                model_type="sentence_transformers",
                batch_size=self.embedding_config.batch_size,
                normalize_embeddings=self.embedding_config.normalize_embeddings,
                cache_enabled=False  # 禁用缓存以强制重新生成
            )
            
            self.embedding_config = new_config
            self.embedding_manager = EmbeddingManager(new_config)
            self.embedding_manager.initialize()
            
            # 重新创建集合
            self.collection = self.client.get_or_create_collection(
                name="enhanced_ai_papers_v2",
                metadata={"hnsw:space": "cosine"}
            )
            
            # 重新添加文档
            self.add_papers_with_metadata(documents, metadatas)
            
            print("✅ 嵌入重建完成")
            
        except Exception as e:
            print(f"❌ 嵌入重建失败: {e}")

# 向后兼容性 - 为了不破坏现有代码
class VectorStore(EnhancedVectorStore):
    """向后兼容的向量存储类"""
    
    def __init__(self, persist_directory="vector_db"):
        # 使用更好的默认模型，但保持向后兼容
        super().__init__(
            persist_directory=persist_directory,
            embedding_model="all-mpnet-base-v2"  # 升级到更好的模型
        )
        
        # 显示升级提示
        print("🔄 已自动升级到增强版嵌入模型")
        print("   - 从 all-MiniLM-L6-v2 (384维) 升级到 all-mpnet-base-v2 (768维)")
        print("   - 预期检索质量显著提升")
    
    def add_papers(self, papers: List[Dict]):
        """兼容原有的添加论文方法"""
        print(f"📝 添加 {len(papers)} 篇论文到增强向量数据库...")
        
        documents = []
        metadatas = []
        
        for paper in papers:
            doc_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            documents.append(doc_text)
            
            metadata = {
                'title': paper['title'],
                'authors': ', '.join(paper['authors']) if isinstance(paper['authors'], list) else str(paper['authors']),
                'published': paper['published'],
                'paper_id': paper['id'],
                'chunk_id': f"{paper['id']}_main"
            }
            metadatas.append(metadata)
        
        # 使用新的方法添加
        self.add_papers_with_metadata(documents, metadatas)
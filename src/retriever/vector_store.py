import chromadb
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch

class VectorStore:
    def __init__(self, persist_directory="vector_db", embedding_model=None):
        """
        基础向量存储 - 确保嵌入模型一致性
        
        Args:
            persist_directory: 存储目录
            embedding_model: 嵌入模型名称，如果不提供则使用系统默认
        """
        from src.config.embedding_config import SystemEmbeddingConfig
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="ai_papers_enhanced",  # 新的collection名称
            metadata={"hnsw:space": "cosine"}
        )
        
        # 使用系统统一的嵌入模型配置
        if embedding_model is None:
            embedding_model = SystemEmbeddingConfig.DEFAULT_MODEL_NAME
        
        # 验证嵌入模型一致性
        if embedding_model != SystemEmbeddingConfig.DEFAULT_MODEL_NAME:
            print(f"⚠️ 警告: 嵌入模型不一致!")
            print(f"   系统默认: {SystemEmbeddingConfig.DEFAULT_MODEL_NAME}")
            print(f"   当前使用: {embedding_model}")
            print("🚨 这可能导致向量空间不一致!")
        
        # 初始化嵌入模型
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.encoder = SentenceTransformer(embedding_model, device=device)
        self.encoder.model_name = embedding_model  # 添加模型名称属性便于验证
        
        print(f"🔥 基础向量存储配置:")
        print(f"   - 嵌入模型: {embedding_model}")
        print(f"   - 计算设备: {device}")
        print(f"   - 向量维度: {self.encoder.get_sentence_embedding_dimension()}")
    
    def add_papers(self, papers: List[Dict]):
        """添加论文到向量数据库（保留原有方法兼容性）"""
        print(f"📝 添加{len(papers)}篇论文到向量数据库...")
        
        documents = []
        metadatas = []
        ids = []
        
        for paper in papers:
            doc_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            documents.append(doc_text)
            
            metadata = {
                'title': paper['title'],
                'authors': ', '.join(paper['authors']) if isinstance(paper['authors'], list) else str(paper['authors']),
                'published': paper['published'],
                'paper_id': paper['id']
            }
            metadatas.append(metadata)
            ids.append(f"paper_{paper['id']}")
        
        # 生成嵌入
        print("🔢 生成向量嵌入...")
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        # 添加到数据库
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print("✅ 论文添加完成")
    
    def add_papers_with_metadata(self, documents: List[str], metadatas: List[Dict]):
        """添加带有增强元数据的文档块"""
        print(f"📝 添加{len(documents)}个增强文档块到向量数据库...")
        
        # 生成唯一ID
        ids = [f"enhanced_chunk_{i}_{metadata.get('chunk_id', 'unknown')}" 
               for i, metadata in enumerate(metadatas)]
        
        # 生成嵌入
        print("🔢 生成向量嵌入...")
        embeddings = self.encoder.encode(documents, show_progress_bar=True)
        
        # 批量添加到数据库
        batch_size = 100  # 分批处理避免内存问题
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            batch_embeddings = embeddings[i:end_idx]
            batch_documents = documents[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            self.collection.add(
                embeddings=batch_embeddings.tolist(),
                documents=batch_documents,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            print(f"✅ 已处理 {end_idx}/{len(documents)} 个文档块")
        
        print("✅ 增强文档块添加完成")
    
    def search(self, query: str, top_k: int = 5, filter_metadata: Dict = None) -> Dict:
        """搜索相关论文（支持元数据过滤）"""
        # 查询向量化
        query_embedding = self.encoder.encode([query])
        
        # 构建搜索参数
        search_params = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        # 添加过滤条件
        if filter_metadata:
            search_params["where"] = filter_metadata
        
        # 搜索
        results = self.collection.query(**search_params)
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def search_by_section(self, query: str, section_type: str, top_k: int = 5) -> Dict:
        """按章节类型搜索"""
        filter_condition = {"section_type": section_type}
        return self.search(query, top_k, filter_condition)
    
    def search_with_formulas(self, query: str, top_k: int = 5) -> Dict:
        """搜索包含公式的内容"""
        filter_condition = {"has_formulas": True}
        return self.search(query, top_k, filter_condition)
    
    def search_with_code(self, query: str, top_k: int = 5) -> Dict:
        """搜索包含代码的内容"""
        filter_condition = {"has_code": True}
        return self.search(query, top_k, filter_condition)

    def search_with_tables(self, query: str, top_k: int = 5) -> Dict:
        """搜索包含表格的内容"""
        filter_condition = {"has_table": True}
        return self.search(query, top_k, filter_condition)

    def search_with_figures(self, query: str, top_k: int = 5) -> Dict:
        """搜索包含图表的内容"""
        filter_condition = {"has_figure": True}
        return self.search(query, top_k, filter_condition)
    
    def get_statistics(self) -> Dict:
        """获取向量数据库统计信息"""
        try:
            all_data = self.collection.get()
            total_docs = len(all_data['documents'])
            
            if total_docs == 0:
                return {"total_documents": 0, "statistics": "空数据库"}
            
            # 统计元数据信息
            metadatas = all_data['metadatas']
            
            # 章节类型分布
            section_distribution = {}
            papers_distribution = {}
            content_features = {"has_formulas": 0, "has_code": 0, "has_citations": 0, "has_table": 0, "has_figure": 0}
            
            for metadata in metadatas:
                # 章节分布
                section_type = metadata.get('section_type', 'unknown')
                section_distribution[section_type] = section_distribution.get(section_type, 0) + 1
                
                # 论文分布
                paper_id = metadata.get('paper_id', 'unknown')
                papers_distribution[paper_id] = papers_distribution.get(paper_id, 0) + 1
                
                # 内容特征
                for feature in content_features:
                    if metadata.get(feature, False):
                        content_features[feature] += 1
            
            return {
                "total_documents": total_docs,
                "unique_papers": len(papers_distribution),
                "section_distribution": section_distribution,
                "content_features": content_features,
                "avg_chunks_per_paper": total_docs / len(papers_distribution) if papers_distribution else 0
            }
            
        except Exception as e:
            return {"error": f"统计信息获取失败: {e}"}
    
    def clear_database(self):
        """清空数据库"""
        try:
            self.client.delete_collection("ai_papers_enhanced")
            self.collection = self.client.get_or_create_collection(
                name="ai_papers_enhanced",
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ 数据库已清空")
        except Exception as e:
            print(f"❌ 清空数据库失败: {e}")

if __name__ == "__main__":
    # 测试向量存储系统
    vs = VectorStore()
    print("向量存储系统测试完成")
    
    # 显示统计信息
    stats = vs.get_statistics()
    print(f"数据库统计: {stats}")
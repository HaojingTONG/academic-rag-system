import chromadb
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import torch

class VectorStore:
    def __init__(self, persist_directory="vector_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="ai_papers",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 初始化嵌入模型
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        print(f"🔥 向量存储使用设备: {device}")
    
    def add_papers(self, papers: List[Dict]):
        """添加论文到向量数据库"""
        print(f"📝 添加{len(papers)}篇论文到向量数据库...")
        
        # 准备文档和元数据
        documents = []
        metadatas = []
        ids = []
        
        for paper in papers:
            # 使用摘要作为主要文档
            doc_text = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
            documents.append(doc_text)
            
            # 元数据
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
    
    def search(self, query: str, top_k: int = 5) -> Dict:
        """搜索相关论文"""
        # 查询向量化
        query_embedding = self.encoder.encode([query])
        
        # 搜索
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }

if __name__ == "__main__":
    # 测试向量存储
    vs = VectorStore()
    print("向量存储系统测试完成")

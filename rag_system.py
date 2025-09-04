#!/usr/bin/env python3
"""
完整的RAG系统
"""

import sys
import json
from pathlib import Path

# 添加src到路径
sys.path.append('src')

from processor.paper_processor import PaperProcessor
from retriever.vector_store import VectorStore
from typing import Dict

class SimpleRAGSystem:
    def __init__(self):
        self.processor = PaperProcessor()
        self.vector_store = VectorStore()
        self.papers_loaded = False
    
    def setup_knowledge_base(self):
        """设置知识库"""
        print("🔧 设置知识库...")
        
        # 检查是否有论文数据
        papers_info_path = "data/papers_info.json"
        if not Path(papers_info_path).exists():
            print("❌ 请先运行 'python simple_demo.py' 收集论文")
            return False
        
        # 处理论文
        print("📄 处理论文...")
        papers = self.processor.process_papers(papers_info_path)
        
        if not papers:
            print("❌ 没有可用的论文数据")
            return False
        
        # 添加到向量数据库
        self.vector_store.add_papers(papers)
        self.papers_loaded = True
        
        print("✅ 知识库设置完成")
        return True
    
    def query(self, question: str) -> Dict:
        """查询系统"""
        if not self.papers_loaded:
            return {"error": "知识库未加载，请先运行setup_knowledge_base()"}
        
        print(f"🔍 搜索: {question}")
        
        # 检索相关文档
        search_results = self.vector_store.search(question, top_k=3)
        
        # 构建回答（简化版，暂时不使用LLM）
        answer_parts = ["基于检索到的论文内容：\n"]
        
        for i, (doc, metadata, distance) in enumerate(zip(
            search_results['documents'],
            search_results['metadatas'], 
            search_results['distances']
        )):
            answer_parts.append(f"\n📄 论文 {i+1}: {metadata['title']}")
            answer_parts.append(f"👥 作者: {metadata['authors']}")
            answer_parts.append(f"📅 发表: {metadata['published']}")
            answer_parts.append(f"🎯 相关度: {1-distance:.3f}")
            answer_parts.append(f"📝 内容摘要: {doc[:300]}...\n")
        
        return {
            "question": question,
            "answer": "\n".join(answer_parts),
            "sources": search_results['metadatas']
        }

def main():
    """主程序"""
    print("🤖 启动简化版RAG系统")
    print("="*50)
    
    # 初始化系统
    rag = SimpleRAGSystem()
    
    # 设置知识库
    if not rag.setup_knowledge_base():
        return
    
    # 交互式查询
    print("\n💬 现在你可以开始提问了！")
    print("输入 'quit' 退出\n")
    
    while True:
        try:
            question = input("🔍 请输入你的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not question:
                continue
            
            # 查询
            result = rag.query(question)
            
            if "error" in result:
                print(f"❌ {result['error']}")
                continue
            
            print(f"\n💡 回答:\n{result['answer']}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 出现错误: {e}")

if __name__ == "__main__":
    main()

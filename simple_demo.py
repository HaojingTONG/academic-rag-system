#!/usr/bin/env python3
"""
简单的演示程序 - 收集论文并测试基本功能
"""

import arxiv
import os
import requests
from pathlib import Path
from tqdm import tqdm
import time

def collect_sample_papers(num_papers=5):
    """收集示例论文"""
    print(f"📚 开始收集{num_papers}篇AI论文...")
    
    # 创建存储目录
    pdf_dir = Path("data/raw_papers")
    pdf_dir.mkdir(exist_ok=True)
    
    # 搜索论文
    client = arxiv.Client()
    search = arxiv.Search(
        query="cat:cs.AI AND (transformer OR attention OR BERT)",
        max_results=num_papers,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    papers = []
    for i, result in enumerate(tqdm(client.results(search), desc="收集论文")):
        paper_data = {
            "id": result.entry_id.split('/')[-1],
            "title": result.title,
            "authors": [str(author) for author in result.authors],
            "abstract": result.summary,
            "published": result.published.strftime("%Y-%m-%d"),
            "pdf_url": result.pdf_url
        }
        
        # 下载PDF
        pdf_path = pdf_dir / f"{paper_data['id']}.pdf"
        try:
            if not pdf_path.exists():
                print(f"📄 下载: {paper_data['title'][:50]}...")
                response = requests.get(result.pdf_url, timeout=30)
                response.raise_for_status()
                
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                
                paper_data["local_path"] = str(pdf_path)
                time.sleep(1)  # 避免请求过快
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            continue
        
        papers.append(paper_data)
        
        if len(papers) >= num_papers:
            break
    
    print(f"✅ 成功收集{len(papers)}篇论文")
    
    # 保存论文信息
    import json
    with open("data/papers_info.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    
    return papers

def test_embedding():
    """测试嵌入功能"""
    print("\n🔢 测试嵌入模型...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        # 加载模型
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        
        # 测试文本
        test_texts = [
            "Transformer architecture with attention mechanism",
            "Deep learning neural networks",
            "Natural language processing"
        ]
        
        # 生成嵌入
        embeddings = model.encode(test_texts)
        print(f"✅ 嵌入维度: {embeddings.shape}")
        print(f"✅ 使用设备: {device}")
        
        return True
    except Exception as e:
        print(f"❌ 嵌入测试失败: {e}")
        return False

def test_ollama():
    """测试Ollama连接"""
    print("\n🤖 测试本地LLM...")
    
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Ollama连接成功")
            print("已安装的模型:")
            print(result.stdout)
            return True
        else:
            print("❌ Ollama连接失败")
            return False
    except Exception as e:
        print(f"❌ Ollama测试失败: {e}")
        return False

def main():
    """主演示程序"""
    print("🧪 开始系统功能测试...")
    print("="*50)
    
    # 1. 收集论文
    papers = collect_sample_papers(5)
    
    # 2. 测试嵌入
    embedding_ok = test_embedding()
    
    # 3. 测试LLM
    llm_ok = test_ollama()
    
    # 结果总结
    print("\n" + "="*50)
    print("📊 测试结果总结:")
    print(f"📚 论文收集: {'✅' if len(papers) > 0 else '❌'} ({len(papers)}篇)")
    print(f"🔢 嵌入模型: {'✅' if embedding_ok else '❌'}")
    print(f"🤖 本地LLM: {'✅' if llm_ok else '❌'}")
    
    if len(papers) > 0 and embedding_ok:
        print("\n🎉 基础功能测试通过！")
        print("下一步可以运行完整的RAG系统")
    else:
        print("\n⚠️ 部分功能需要修复")

if __name__ == "__main__":
    main()

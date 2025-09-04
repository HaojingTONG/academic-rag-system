import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import Dict, List

class PaperProcessor:
    def __init__(self):
        self.processed_data = {}
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """提取PDF文本"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"PDF处理错误: {e}")
            return ""
    
    def extract_abstract(self, text: str) -> str:
        """提取摘要"""
        # 简单的摘要提取
        abstract_pattern = r"Abstract\s*\n(.*?)(?:\n\n|\n1\.|\nIntroduction)"
        match = re.search(abstract_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 如果没找到，返回前500个字符
        return text[:500] + "..." if len(text) > 500 else text
    
    def process_papers(self, papers_info_path: str) -> List[Dict]:
        """处理所有论文"""
        import json
        
        with open(papers_info_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        processed_papers = []
        for paper in papers:
            if 'local_path' in paper and Path(paper['local_path']).exists():
                # 提取PDF文本
                full_text = self.extract_pdf_text(paper['local_path'])
                
                # 提取摘要
                extracted_abstract = self.extract_abstract(full_text)
                
                processed_paper = {
                    'id': paper['id'],
                    'title': paper['title'],
                    'authors': paper['authors'],
                    'abstract': paper['abstract'],  # 原始摘要
                    'extracted_abstract': extracted_abstract,  # 从PDF提取的摘要
                    'full_text': full_text[:5000],  # 前5000字符
                    'published': paper['published']
                }
                
                processed_papers.append(processed_paper)
        
        return processed_papers

if __name__ == "__main__":
    processor = PaperProcessor()
    papers = processor.process_papers("data/papers_info.json")
    print(f"处理了{len(papers)}篇论文")

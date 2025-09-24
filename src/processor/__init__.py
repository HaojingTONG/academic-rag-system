"""
数据处理模块

本模块包含了AI论文RAG系统的核心数据处理组件：
- 智能分块策略
- 多模态内容提取  
- 元数据增强
- 质量过滤和去重
"""

from .pdf_processor import AcademicPDFProcessor
from .document_chunker import DocumentChunker

__version__ = "0.1.0"

__all__ = [
    'AcademicPDFProcessor',
    'DocumentChunker'
]

class DataProcessor:
    """数据处理统一接口"""

    def __init__(self):
        self.pdf_processor = AcademicPDFProcessor()
        self.chunker = DocumentChunker()

    def process_paper(self, paper_data: dict, pdf_path: str = None) -> dict:
        """处理单篇论文的完整流程"""

        print(f"🔄 开始处理论文: {paper_data.get('title', 'Unknown')[:50]}...")

        # 1. 提取PDF内容
        if pdf_path:
            pdf_content = self.pdf_processor.extract_paper_content(pdf_path)
            paper_data.update(pdf_content)

        # 2. 智能分块
        full_text = paper_data.get('text_content', '') or paper_data.get('full_text', '')
        if full_text:
            chunks = self.chunker.chunk_document(full_text, paper_data.get('id', 'unknown'))
            paper_data['processed_chunks'] = chunks
            paper_data['chunk_count'] = len(chunks)

        print(f"✅ 论文处理完成: {paper_data.get('chunk_count', 0)} 个文档块")

        return paper_data

def process_single_paper(paper_data: dict, pdf_path: str = None) -> dict:
    """处理单篇论文的便捷函数"""
    processor = DataProcessor()
    return processor.process_paper(paper_data, pdf_path)

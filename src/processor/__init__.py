"""
数据处理模块

本模块包含了AI论文RAG系统的核心数据处理组件：
- 智能分块策略
- 多模态内容提取  
- 元数据增强
- 质量过滤和去重
"""

from .intelligent_chunker import IntelligentChunker, ChunkMetadata
from .multimodal_extractor import MultimodalExtractor
from .metadata_enricher import MetadataEnricher
from .quality_filter import QualityFilter

__version__ = "0.1.0"

__all__ = [
    'IntelligentChunker',
    'ChunkMetadata', 
    'MultimodalExtractor',
    'MetadataEnricher',
    'QualityFilter',
    'DataProcessor'
]

class DataProcessor:
    """数据处理统一接口"""
    
    def __init__(self):
        self.chunker = IntelligentChunker()
        self.multimodal_extractor = MultimodalExtractor() 
        self.metadata_enricher = MetadataEnricher()
        self.quality_filter = QualityFilter()
    
    def process_paper(self, paper_data: dict, pdf_path: str = None) -> dict:
        """处理单篇论文的完整流程"""
        
        print(f"🔄 开始处理论文: {paper_data.get('title', 'Unknown')[:50]}...")
        
        # 1. 提取多模态内容
        if pdf_path:
            multimodal_content = self.multimodal_extractor.extract_multimodal_content(pdf_path)
            paper_data.update(multimodal_content)
        
        # 2. 增强元数据
        enriched_paper = self.metadata_enricher.enrich_metadata(paper_data)
        
        # 3. 智能分块
        full_text = enriched_paper.get('text_content', '') or enriched_paper.get('full_text', '')
        if full_text:
            chunks = self.chunker.chunk_paper(full_text, enriched_paper['id'])
            
            # 4. 质量过滤
            filtered_chunks = self.quality_filter.filter_chunks(chunks)
            
            enriched_paper['processed_chunks'] = filtered_chunks
            enriched_paper['chunk_count'] = len(filtered_chunks)
        
        print(f"✅ 论文处理完成: {enriched_paper.get('chunk_count', 0)} 个高质量文档块")
        
        return enriched_paper

def process_single_paper(paper_data: dict, pdf_path: str = None) -> dict:
    """处理单篇论文的便捷函数"""
    processor = DataProcessor()
    return processor.process_paper(paper_data, pdf_path)

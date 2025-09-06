# src/processor/intelligent_chunker.py
import re
import nltk
from typing import List, Dict, Tuple
from dataclasses import dataclass
import spacy

@dataclass
class ChunkMetadata:
    chunk_id: str
    section_type: str  # abstract, introduction, method, results, conclusion
    paragraph_index: int
    sentence_count: int
    word_count: int
    has_formulas: bool
    has_code: bool
    has_citations: bool

class IntelligentChunker:
    def __init__(self):
        # 下载必要的NLTK数据
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # 加载spaCy模型用于句子分割
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("请安装spaCy英文模型: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def chunk_paper(self, paper_text: str, paper_id: str, 
                   max_chunk_size: int = 1000, 
                   overlap_size: int = 200) -> List[Dict]:
        """智能分块论文内容"""
        
        # 1. 预处理文本
        cleaned_text = self._preprocess_text(paper_text)
        
        # 2. 检测论文结构
        sections = self._detect_paper_sections(cleaned_text)
        
        # 3. 基于语义的分块
        chunks = []
        for section_name, section_text in sections.items():
            section_chunks = self._semantic_chunking(
                section_text, section_name, paper_id, 
                max_chunk_size, overlap_size
            )
            chunks.extend(section_chunks)
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 移除过多的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 修复常见的PDF解析错误
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # 添加缺失的空格
        text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)     # 句子间添加空格
        
        # 处理连字符
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        return text.strip()
    
    def _detect_paper_sections(self, text: str) -> Dict[str, str]:
        """检测论文章节结构"""
        sections = {}
        
        # 定义章节模式
        section_patterns = {
            'abstract': r'(abstract|summary)\s*\n(.*?)(?=\n\s*(?:introduction|1\.|keywords|\n\n))',
            'introduction': r'(introduction|1\.?\s*introduction)\s*\n(.*?)(?=\n\s*(?:related work|background|2\.|method|\n\n))',
            'method': r'(method|methodology|approach|3\..*?method|model)\s*\n(.*?)(?=\n\s*(?:experiment|result|4\.|evaluation|\n\n))',
            'results': r'(results?|experiments?|evaluation|4\..*?result)\s*\n(.*?)(?=\n\s*(?:discussion|conclusion|5\.|\n\n))',
            'conclusion': r'(conclusion|discussion|5\..*?conclusion)\s*\n(.*?)(?=\n\s*(?:reference|acknowledgment|appendix|\n\n))',
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text.lower(), re.DOTALL | re.IGNORECASE)
            if match:
                sections[section_name] = match.group(2).strip()
        
        # 如果无法检测到结构，将整个文本作为一个章节
        if not sections:
            sections['full_text'] = text
        
        return sections
    
    def _semantic_chunking(self, text: str, section_type: str, paper_id: str,
                          max_size: int, overlap: int) -> List[Dict]:
        """基于语义的分块"""
        chunks = []
        
        # 按段落分割
        paragraphs = self._split_by_paragraphs(text)
        
        current_chunk = ""
        chunk_index = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            # 检查是否需要创建新块
            if len(current_chunk) + len(paragraph) > max_size and current_chunk:
                # 创建当前块
                chunk_data = self._create_chunk(
                    current_chunk, paper_id, section_type, 
                    chunk_index, para_idx
                )
                chunks.append(chunk_data)
                
                # 开始新块（保留重叠）
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + paragraph
                chunk_index += 1
            else:
                current_chunk += "\n" + paragraph if current_chunk else paragraph
        
        # 处理最后一个块
        if current_chunk:
            chunk_data = self._create_chunk(
                current_chunk, paper_id, section_type, 
                chunk_index, len(paragraphs)
            )
            chunks.append(chunk_data)
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 按双换行符分割段落
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 过滤空段落和过短段落
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 50]
        
        return paragraphs
    
    def _create_chunk(self, text: str, paper_id: str, section_type: str,
                     chunk_index: int, para_index: int) -> Dict:
        """创建分块数据"""
        # 分析文本特征
        sentences = self._split_sentences(text)
        word_count = len(text.split())
        
        # 检测特殊内容
        has_formulas = bool(re.search(r'\$.*?\$|\\[a-zA-Z]+', text))
        has_code = bool(re.search(r'```|def |class |import |from ', text))
        has_citations = bool(re.search(r'\[[0-9,\-\s]+\]|\([A-Za-z]+,?\s*[0-9]{4}\)', text))
        
        metadata = ChunkMetadata(
            chunk_id=f"{paper_id}_chunk_{chunk_index}",
            section_type=section_type,
            paragraph_index=para_index,
            sentence_count=len(sentences),
            word_count=word_count,
            has_formulas=has_formulas,
            has_code=has_code,
            has_citations=has_citations
        )
        
        return {
            'text': text,
            'metadata': metadata.__dict__,
            'paper_id': paper_id,
            'chunk_index': chunk_index
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # 使用NLTK作为备选
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """获取重叠文本"""
        if len(text) <= overlap_size:
            return text
        
        # 尝试在句子边界截取
        sentences = self._split_sentences(text)
        overlap_text = ""
        
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= overlap_size:
                overlap_text = sentence + " " + overlap_text
            else:
                break
        
        return overlap_text.strip()


# 使用示例
def test_intelligent_chunker():
    chunker = IntelligentChunker()
    
    sample_text = """
    Abstract
    This paper presents a novel approach to natural language processing...
    
    1. Introduction
    Natural language processing has seen significant advances in recent years...
    
    2. Method
    Our proposed method consists of three main components...
    """
    
    chunks = chunker.chunk_paper(sample_text, "test_paper", max_chunk_size=500)
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:")
        print(f"Section: {chunk['metadata']['section_type']}")
        print(f"Word count: {chunk['metadata']['word_count']}")
        print(f"Text: {chunk['text'][:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    test_intelligent_chunker()
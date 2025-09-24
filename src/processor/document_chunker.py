# src/processor/document_chunker.py
"""
文档切分模块 - 实现多种文档切分策略
支持固定长度、语义分割、重叠切分等多种策略
"""

import re
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Chunk:
    """文档块数据结构"""
    text: str                    # 文本内容
    chunk_id: str               # 块ID
    paper_id: str               # 论文ID
    chunk_index: int            # 块索引
    start_char: int             # 在原文中的起始位置
    end_char: int               # 在原文中的结束位置
    metadata: Dict              # 元数据
    section_type: str = "content"  # 章节类型
    word_count: int = 0         # 词数
    char_count: int = 0         # 字符数

    def __post_init__(self):
        """自动计算统计信息"""
        self.char_count = len(self.text)
        self.word_count = len(self.text.split())

@dataclass
class ChunkingConfig:
    """切分配置"""
    strategy: str = "fixed_size"        # 切分策略
    chunk_size: int = 500              # 块大小（字符数）
    chunk_overlap: int = 50            # 重叠大小
    min_chunk_size: int = 100          # 最小块大小
    max_chunk_size: int = 1000         # 最大块大小
    preserve_paragraphs: bool = True    # 是否保持段落完整性
    preserve_sentences: bool = True     # 是否保持句子完整性
    section_aware: bool = True         # 是否感知论文章节结构

class BaseChunker(ABC):
    """切分器基类"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        
    @abstractmethod
    def chunk_document(self, text: str, paper_id: str, metadata: Dict = None) -> List[Chunk]:
        """切分文档"""
        pass
    
    def _create_chunk(self, text: str, paper_id: str, chunk_index: int, 
                     start_char: int, end_char: int, metadata: Dict = None) -> Chunk:
        """创建文档块"""
        chunk_id = f"{paper_id}_chunk_{chunk_index}"
        
        # 增强元数据
        enhanced_metadata = {
            'paper_id': paper_id,
            'chunk_index': chunk_index,
            'start_char': start_char,
            'end_char': end_char,
            'chunking_strategy': self.config.strategy,
            'chunk_size_config': self.config.chunk_size,
            'overlap_config': self.config.chunk_overlap,
        }
        
        if metadata:
            # 处理metadata，确保所有值都是ChromaDB支持的类型
            for key, value in metadata.items():
                if isinstance(value, list):
                    # 将列表转换为字符串
                    enhanced_metadata[key] = ', '.join(str(v) for v in value) if value else ""
                elif isinstance(value, dict):
                    # 跳过字典类型或转换为字符串
                    enhanced_metadata[f"{key}_str"] = str(value)
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    enhanced_metadata[key] = value
                else:
                    # 其他类型转换为字符串
                    enhanced_metadata[key] = str(value)
        
        # 分析内容特征
        enhanced_metadata.update(self._analyze_content_features(text))
        
        return Chunk(
            text=text,
            chunk_id=chunk_id,
            paper_id=paper_id,
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata=enhanced_metadata
        )
    
    def _analyze_content_features(self, text: str) -> Dict:
        """分析内容特征"""
        return {
            'has_formulas': bool(re.search(r'\$.*?\$|\\[a-zA-Z]+|\b(?:equation|formula)\b', text, re.IGNORECASE)),
            'has_code': bool(re.search(r'def |class |import |function|\{.*\}|```', text, re.IGNORECASE)),
            'has_citations': bool(re.search(r'\[[0-9,\-\s]+\]|\([A-Za-z]+,?\s*[0-9]{4}\)', text)),
            'has_numbers': bool(re.search(r'\b\d+\.?\d*\b', text)),
            'has_urls': bool(re.search(r'http[s]?://\S+|www\.\S+', text)),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'sentence_count': len(re.findall(r'[.!?]+', text))
        }

class FixedSizeChunker(BaseChunker):
    """固定大小切分器"""
    
    def chunk_document(self, text: str, paper_id: str, metadata: Dict = None) -> List[Chunk]:
        """按固定大小切分文档"""
        chunks = []
        text_length = len(text)
        chunk_index = 0
        
        start = 0
        while start < text_length:
            end = min(start + self.config.chunk_size, text_length)
            
            # 如果不是最后一个chunk，尝试在合适的位置切分
            if end < text_length and self.config.preserve_sentences:
                # 寻找句子边界
                sentence_end = self._find_sentence_boundary(text, end)
                if sentence_end > start + self.config.min_chunk_size:
                    end = sentence_end
            
            chunk_text = text[start:end].strip()
            
            # 跳过太短的块
            if len(chunk_text) < self.config.min_chunk_size and chunk_index > 0:
                # 将剩余文本合并到上一个chunk
                if chunks:
                    last_chunk = chunks[-1]
                    combined_text = last_chunk.text + " " + chunk_text
                    chunks[-1] = self._create_chunk(
                        combined_text, paper_id, last_chunk.chunk_index,
                        last_chunk.start_char, end, metadata
                    )
                break
            
            if chunk_text:
                chunk = self._create_chunk(chunk_text, paper_id, chunk_index, start, end, metadata)
                chunks.append(chunk)
                chunk_index += 1
            
            # 计算下一个块的起始位置（考虑重叠）
            start = max(start + 1, end - self.config.chunk_overlap)
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """寻找句子边界"""
        # 向后寻找句号、问号、感叹号
        for i in range(position, min(position + 100, len(text))):
            if text[i] in '.!?':
                # 确保不是缩写或数字中的点
                if i + 1 < len(text) and text[i + 1].isspace():
                    return i + 1
        
        # 向前寻找句子边界
        for i in range(position, max(position - 100, 0), -1):
            if text[i] in '.!?' and i + 1 < len(text) and text[i + 1].isspace():
                return i + 1
        
        return position

class SemanticChunker(BaseChunker):
    """语义感知切分器"""
    
    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.section_patterns = {
            'abstract': r'abstract\s*:?|摘\s*要',
            'introduction': r'introduction|引\s*言|前\s*言',
            'method': r'method|methodology|approach|方\s*法',
            'experiment': r'experiment|evaluation|实\s*验|评\s*估',
            'result': r'result|findings|结\s*果',
            'discussion': r'discussion|analysis|讨\s*论|分\s*析',
            'conclusion': r'conclusion|结\s*论',
            'reference': r'reference|bibliography|参\s*考\s*文\s*献'
        }
    
    def chunk_document(self, text: str, paper_id: str, metadata: Dict = None) -> List[Chunk]:
        """基于语义结构切分文档"""
        # 首先尝试识别章节
        sections = self._identify_sections(text)
        
        if not sections or len(sections) == 1:
            # 如果无法识别章节，回退到段落切分
            return self._chunk_by_paragraphs(text, paper_id, metadata)
        
        chunks = []
        chunk_index = 0
        
        for section_type, section_text, start_pos, end_pos in sections:
            # 如果章节太长，进一步切分
            if len(section_text) > self.config.max_chunk_size:
                section_chunks = self._chunk_long_section(
                    section_text, paper_id, chunk_index, start_pos, section_type, metadata
                )
                chunks.extend(section_chunks)
                chunk_index += len(section_chunks)
            else:
                # 创建章节级别的chunk
                chunk = self._create_chunk(
                    section_text, paper_id, chunk_index, start_pos, end_pos, metadata
                )
                chunk.section_type = section_type
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _identify_sections(self, text: str) -> List[Tuple[str, str, int, int]]:
        """识别文档章节"""
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content = []
        current_start = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip().lower()
            
            # 检查是否是新章节标题
            section_type = None
            for section, pattern in self.section_patterns.items():
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    section_type = section
                    break
            
            if section_type:
                # 保存前一个章节
                if current_section and current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append((
                            current_section, 
                            content, 
                            current_start,
                            current_start + len(content)
                        ))
                
                # 开始新章节
                current_section = section_type
                current_content = [line]
                current_start = text.find('\n'.join(lines[:i])) if i > 0 else 0
            else:
                # 添加到当前章节
                if current_section:
                    current_content.append(line)
        
        # 添加最后一个章节
        if current_section and current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append((
                    current_section, 
                    content, 
                    current_start,
                    current_start + len(content)
                ))
        
        return sections
    
    def _chunk_by_paragraphs(self, text: str, paper_id: str, metadata: Dict = None) -> List[Chunk]:
        """按段落切分"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        chunk_index = 0
        current_chunk = []
        current_size = 0
        start_pos = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            # 如果加上这个段落会超过最大大小，先保存当前chunk
            if current_chunk and current_size + para_size > self.config.chunk_size:
                chunk_text = '\n\n'.join(current_chunk)
                chunk = self._create_chunk(
                    chunk_text, paper_id, chunk_index, 
                    start_pos, start_pos + len(chunk_text), metadata
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # 重叠处理：保留最后一个段落
                if self.config.chunk_overlap > 0 and len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1], paragraph]
                    current_size = len(current_chunk[-2]) + para_size
                else:
                    current_chunk = [paragraph]
                    current_size = para_size
                
                start_pos = text.find(paragraph)
            else:
                current_chunk.append(paragraph)
                current_size += para_size
        
        # 处理最后一个chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk = self._create_chunk(
                chunk_text, paper_id, chunk_index,
                start_pos, start_pos + len(chunk_text), metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_long_section(self, section_text: str, paper_id: str, start_chunk_index: int,
                           start_pos: int, section_type: str, metadata: Dict = None) -> List[Chunk]:
        """切分长章节"""
        # 使用固定大小切分器处理长章节
        temp_chunker = FixedSizeChunker(self.config)
        chunks = temp_chunker.chunk_document(section_text, paper_id, metadata)
        
        # 更新chunk索引和章节类型
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = start_chunk_index + i
            chunk.chunk_id = f"{paper_id}_chunk_{chunk.chunk_index}"
            chunk.section_type = section_type
            chunk.start_char += start_pos
            chunk.end_char += start_pos
        
        return chunks

class HybridChunker(BaseChunker):
    """混合切分器 - 结合多种策略"""
    
    def chunk_document(self, text: str, paper_id: str, metadata: Dict = None) -> List[Chunk]:
        """使用混合策略切分文档"""
        # 首先尝试语义切分
        semantic_chunker = SemanticChunker(self.config)
        chunks = semantic_chunker.chunk_document(text, paper_id, metadata)
        
        # 检查chunk质量，对不合适的chunk进行再切分
        refined_chunks = []
        for chunk in chunks:
            if len(chunk.text) > self.config.max_chunk_size:
                # 对过长的chunk进行固定大小切分
                fixed_chunker = FixedSizeChunker(self.config)
                sub_chunks = fixed_chunker.chunk_document(chunk.text, paper_id, metadata)
                
                # 更新子chunk的索引和位置
                base_index = len(refined_chunks)
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.chunk_index = base_index + i
                    sub_chunk.chunk_id = f"{paper_id}_chunk_{sub_chunk.chunk_index}"
                    sub_chunk.start_char += chunk.start_char
                    sub_chunk.end_char += chunk.start_char
                    sub_chunk.section_type = chunk.section_type
                
                refined_chunks.extend(sub_chunks)
            else:
                refined_chunks.append(chunk)
        
        return refined_chunks

class DocumentChunker:
    """文档切分器主类"""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.chunkers = {
            'fixed_size': FixedSizeChunker,
            'semantic': SemanticChunker,
            'hybrid': HybridChunker
        }
    
    def chunk_document(self, text: str, paper_id: str, metadata: Dict = None) -> List[Chunk]:
        """切分文档"""
        if self.config.strategy not in self.chunkers:
            raise ValueError(f"未知的切分策略: {self.config.strategy}")
        
        chunker_class = self.chunkers[self.config.strategy]
        chunker = chunker_class(self.config)
        
        chunks = chunker.chunk_document(text, paper_id, metadata)
        
        # 后处理：质量检查和优化
        chunks = self._post_process_chunks(chunks)
        
        return chunks
    
    def _post_process_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """后处理chunks"""
        processed_chunks = []
        
        for chunk in chunks:
            # 清理文本
            cleaned_text = self._clean_text(chunk.text)
            if len(cleaned_text.strip()) < self.config.min_chunk_size:
                continue
            
            # 更新chunk
            chunk.text = cleaned_text
            chunk.char_count = len(cleaned_text)
            chunk.word_count = len(cleaned_text.split())
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_text(self, text: str) -> str:
        """
        改进的文本清洗，保留换行与结构
        保留段内换行、列表格式、章节结构，同时去除重复空格与噪声
        """
        if not text or not text.strip():
            return ""

        # 第1步：预处理 - 标准化行结束符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 第2步：按行处理，保留重要结构
        lines = text.split('\n')
        processed_lines = []

        for i, line in enumerate(lines):
            processed_line = self._process_line_for_cleaning(line, i, lines)
            if processed_line is not None:  # None表示跳过该行
                processed_lines.append(processed_line)

        # 第3步：智能合并行，保留段落结构
        structured_text = self._merge_lines_intelligently(processed_lines)

        # 第4步：最终清理
        final_text = self._final_text_cleanup(structured_text)

        return final_text.strip()

    def _process_line_for_cleaning(self, line: str, line_index: int, all_lines: list) -> str:
        """处理单行文本，用于文本清洗"""
        # 跳过完全空白的行（但保留单个换行作为段落分隔）
        if not line.strip():
            return ""

        # 检查是否是噪声行
        if self._is_noise_line_for_cleaning(line):
            return None  # 跳过噪声行

        # 检查是否是列表项
        if self._is_list_item_for_cleaning(line):
            return self._clean_list_item_for_cleaning(line)

        # 检查是否是结构化行（章节标题等）
        if self._is_structure_line_for_cleaning(line):
            return self._clean_structure_line_for_cleaning(line)

        # 普通文本行清理
        return self._clean_normal_line_for_cleaning(line)

    def _is_noise_line_for_cleaning(self, line: str) -> bool:
        """检查是否是噪声行"""
        line_stripped = line.strip()

        if not line_stripped:
            return False

        # 噪声模式
        noise_patterns = [
            r'^\s*\d+\s*$',                    # 单独的页码
            r'^\s*Figure\s+\d+[:\.]?\s*$',     # 单独的图片标题行
            r'^\s*Table\s+\d+[:\.]?\s*$',      # 单独的表格标题行
            r'^\s*Fig\.\s*\d+\s*$',            # 简化图片引用
            r'^\s*[^\w\s]{3,}\s*$',            # 纯符号行(如分隔线)
        ]

        for pattern in noise_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True

        # 长度过短且没有实际内容的行
        if len(line_stripped) < 3 and not re.search(r'\w', line_stripped):
            return True

        return False

    def _is_list_item_for_cleaning(self, line: str) -> bool:
        """检查是否是列表项"""
        list_patterns = [
            r'^\s*[-•·]\s+',           # 项目符号列表
            r'^\s*[0-9]+\.\s+',        # 数字列表 (1. 2. 3.)
            r'^\s*[0-9]+\)\s+',        # 数字列表 (1) 2) 3))
            r'^\s*[a-zA-Z]\.\s+',      # 字母列表 (a. b. c.)
            r'^\s*[a-zA-Z]\)\s+',      # 字母列表 (a) b) c))
            r'^\s*[ivx]+\.\s+',        # 罗马数字列表 (i. ii. iii.)
            r'^\s*\*\s+',              # 星号列表
            r'^\s*\+\s+',              # 加号列表
        ]

        for pattern in list_patterns:
            if re.match(pattern, line):
                return True
        return False

    def _is_structure_line_for_cleaning(self, line: str) -> bool:
        """检查是否是结构化行（章节标题等）"""
        line_stripped = line.strip()

        # 章节标题模式
        if re.match(r'^\s*(?:\d+\.?\s*)?[A-Z][^.!?]*$', line_stripped):
            return True

        # 包含公式、引用等特殊内容
        special_patterns = [
            r'\$.*?\$|\\[a-zA-Z]+\{.*?\}',              # 数学公式
            r'\[[0-9,\-\s]+\]|\([A-Za-z]+,?\s*[0-9]{4}\)',  # 引用
            r'http[s]?://\S+|www\.\S+',                     # URL
            r'```[\s\S]*?```|`[^`]+`',                # 代码块
        ]

        for pattern in special_patterns:
            if re.search(pattern, line):
                return True

        return False

    def _clean_list_item_for_cleaning(self, line: str) -> str:
        """清理列表项"""
        list_patterns = [
            r'^\s*[-•·]\s+',           # 项目符号列表
            r'^\s*[0-9]+\.\s+',        # 数字列表
            r'^\s*[0-9]+\)\s+',        # 数字列表
            r'^\s*[a-zA-Z]\.\s+',      # 字母列表
            r'^\s*[a-zA-Z]\)\s+',      # 字母列表
            r'^\s*[ivx]+\.\s+',        # 罗马数字列表
            r'^\s*\*\s+',              # 星号列表
            r'^\s*\+\s+',              # 加号列表
        ]

        # 找到列表标识符
        for pattern in list_patterns:
            match = re.match(pattern, line)
            if match:
                prefix = match.group(0)  # 列表标识符部分
                content = line[match.end():].strip()  # 内容部分

                # 清理内容部分
                content = re.sub(r'\s+', ' ', content)

                return prefix + content

        # 如果没有匹配到模式，按普通行处理
        return self._clean_normal_line_for_cleaning(line)

    def _clean_structure_line_for_cleaning(self, line: str) -> str:
        """清理结构化行"""
        # 移除行首尾多余空格，但保留内部格式
        line = line.strip()

        # 清理多余的空格（但不破坏特殊格式）
        line = re.sub(r'[ \t]+', ' ', line)

        return line

    def _clean_normal_line_for_cleaning(self, line: str) -> str:
        """清理普通文本行"""
        # 移除行首尾空格
        line = line.strip()

        # 清理多余的空格和制表符
        line = re.sub(r'[ \t]+', ' ', line)

        # 清理多余的标点符号
        line = re.sub(r'([.!?]){2,}', r'\1', line)  # 重复标点
        line = re.sub(r'([,;:]){2,}', r'\1', line)   # 重复标点

        return line

    def _merge_lines_intelligently(self, lines: list) -> str:
        """智能合并行，保留段落结构"""
        if not lines:
            return ""

        result_lines = []
        i = 0

        while i < len(lines):
            current_line = lines[i]

            # 空行作为段落分隔符
            if current_line == "":
                result_lines.append("")
                i += 1
                continue

            # 结构化行（标题、列表等）单独成行
            if (self._is_structure_line_for_cleaning(current_line) or
                self._is_list_item_for_cleaning(current_line)):
                result_lines.append(current_line)
                i += 1
                continue

            # 普通文本行：检查是否需要与下一行合并
            merged_line = current_line
            j = i + 1

            while j < len(lines):
                next_line = lines[j]

                # 遇到空行、结构化行或列表项，停止合并
                if (next_line == "" or
                    self._is_structure_line_for_cleaning(next_line) or
                    self._is_list_item_for_cleaning(next_line)):
                    break

                # 检查是否应该合并
                if self._should_merge_lines_for_cleaning(merged_line, next_line):
                    merged_line += " " + next_line
                    j += 1
                else:
                    break

            result_lines.append(merged_line)
            i = j

        return '\n'.join(result_lines)

    def _should_merge_lines_for_cleaning(self, line1: str, line2: str) -> bool:
        """判断两行是否应该合并"""
        # 如果第一行以句号、感叹号或问号结尾，通常不合并
        if re.search(r'[.!?]\s*$', line1.strip()):
            return False

        # 如果第二行看起来像新句子的开始（大写字母），可能不应该合并
        if re.match(r'^[A-Z]', line2.strip()):
            # 但如果第一行很短（可能是断行），还是要合并
            return len(line1.strip()) < 50

        # 如果第二行是数字或特殊字符开头，可能不应该合并
        if re.match(r'^\d+|^[^\w\s]', line2.strip()):
            return False

        # 默认情况下合并短行
        return True

    def _final_text_cleanup(self, text: str) -> str:
        """最终清理"""
        # 清理多个连续空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 清理行尾空格
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)

        # 移除特殊字符（但保留基本标点和数学符号）
        # 更温和的清理，避免破坏重要内容
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]\"\'\/\\\$\%\#\@\&\*\+\=\<\>\~\`\n]', ' ', text)

        # 最后清理：移除多余空格，但保留换行
        text = re.sub(r'[ \t]+', ' ', text)

        return text
    
    def get_chunking_stats(self, chunks: List[Chunk]) -> Dict:
        """获取切分统计信息"""
        if not chunks:
            return {}
        
        chunk_sizes = [chunk.char_count for chunk in chunks]
        word_counts = [chunk.word_count for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'avg_word_count': sum(word_counts) / len(word_counts),
            'total_characters': sum(chunk_sizes),
            'total_words': sum(word_counts),
            'section_types': list(set(chunk.section_type for chunk in chunks))
        }

# 使用示例和测试函数
def test_document_chunker():
    """测试文档切分器"""
    
    # 测试文本
    test_text = """
    Title: Attention Is All You Need
    
    Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism.
    
    1. Introduction
    
    Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation.
    
    2. Background
    
    The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block.
    
    3. Model Architecture
    
    Most competitive neural sequence transduction models have an encoder-decoder structure [5]. Here, the encoder maps an input sequence of symbol representations to a sequence of continuous representations.
    """
    
    print("测试文档切分器...")
    
    # 测试不同配置
    configs = [
        ChunkingConfig(strategy="fixed_size", chunk_size=200, chunk_overlap=50),
        ChunkingConfig(strategy="semantic", chunk_size=300, chunk_overlap=30),
        ChunkingConfig(strategy="hybrid", chunk_size=250, chunk_overlap=40)
    ]
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}: {config.strategy}")
        print(f"块大小: {config.chunk_size}, 重叠: {config.chunk_overlap}")
        
        chunker = DocumentChunker(config)
        chunks = chunker.chunk_document(test_text, "test_paper")
        stats = chunker.get_chunking_stats(chunks)
        
        print(f"切分结果:")
        print(f"  总块数: {stats['total_chunks']}")
        print(f"  平均块大小: {stats['avg_chunk_size']:.1f} 字符")
        print(f"  平均词数: {stats['avg_word_count']:.1f}")
        print(f"  章节类型: {stats['section_types']}")
        
        for j, chunk in enumerate(chunks[:3]):  # 显示前3个chunk
            print(f"\nChunk {j+1} (类型: {chunk.section_type}):")
            print(f"  长度: {chunk.char_count} 字符")
            print(f"  内容预览: {chunk.text[:100]}...")

if __name__ == "__main__":
    test_document_chunker()
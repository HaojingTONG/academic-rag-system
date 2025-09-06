# src/processor/quality_filter.py
import re
import hashlib
from typing import List, Dict, Set, Tuple
from difflib import SequenceMatcher
import numpy as np
from collections import Counter

class QualityFilter:
    def __init__(self):
        self.min_word_count = 50
        self.min_sentence_count = 3
        self.max_repetition_ratio = 0.3
        self.similarity_threshold = 0.85
        
        # 低质量文本模式
        self.low_quality_patterns = [
            r'^[\s\d\W]+$',              # 只包含数字、标点和空白
            r'^\s*$',                    # 空文本
            r'^.{0,20}$',               # 过短文本
            r'(.)\1{10,}',              # 重复字符
            r'\b(\w+)\s+\1\s+\1\b',     # 重复单词
        ]
        
        # 噪音模式（页眉、页脚等）
        self.noise_patterns = [
            r'^\d+\s*$',                # 纯页码
            r'^page\s+\d+',             # 页码标记
            r'^figure\s+\d+',           # 图片标记（孤立的）
            r'^table\s+\d+',            # 表格标记（孤立的）
            r'^references?\s*$',        # 孤立的参考文献标题
            r'^\s*[a-z]\)\s*$',         # 列表标记
            r'^\s*\d+\.\s*$',           # 数字标记
        ]
    
    def filter_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """过滤文档块"""
        print(f"开始质量过滤，输入{len(chunks)}个文档块...")
        
        # 第一步：基础质量过滤
        quality_filtered = []
        for chunk in chunks:
            if self._is_good_quality(chunk['text']):
                quality_filtered.append(chunk)
        
        print(f"基础质量过滤后剩余{len(quality_filtered)}个块")
        
        # 第二步：去重
        deduplicated = self._remove_duplicates(quality_filtered)
        print(f"去重后剩余{len(deduplicated)}个块")
        
        # 第三步：去除噪音
        clean_chunks = self._remove_noise(deduplicated)
        print(f"噪音过滤后剩余{len(clean_chunks)}个块")
        
        return clean_chunks
    
    def _is_good_quality(self, text: str) -> bool:
        """判断文本质量"""
        # 基础长度检查
        if len(text.strip()) < 30:
            return False
        
        # 单词和句子数量检查
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        if len(words) < self.min_word_count:
            return False
        
        if len([s for s in sentences if len(s.strip()) > 5]) < self.min_sentence_count:
            return False
        
        # 检查低质量模式
        for pattern in self.low_quality_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
        
        # 重复性检查
        if self._calculate_repetition_ratio(text) > self.max_repetition_ratio:
            return False
        
        # 字符多样性检查
        if self._calculate_character_diversity(text) < 0.1:
            return False
        
        # 语言连贯性检查
        if not self._check_language_coherence(text):
            return False
        
        return True
    
    def _calculate_repetition_ratio(self, text: str) -> float:
        """计算重复率"""
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
        
        word_counts = Counter(words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        
        return repeated_words / len(words)
    
    def _calculate_character_diversity(self, text: str) -> float:
        """计算字符多样性"""
        if len(text) == 0:
            return 0.0
        
        unique_chars = len(set(text.lower()))
        return unique_chars / len(text)
    
    def _check_language_coherence(self, text: str) -> bool:
        """检查语言连贯性"""
        # 检查是否有足够的字母字符
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.6:
            return False
        
        # 检查平均单词长度
        words = [w for w in text.split() if w.isalpha()]
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length < 2 or avg_word_length > 15:
                return False
        
        return True
    
    def _remove_duplicates(self, chunks: List[Dict]) -> List[Dict]:
        """去除重复内容"""
        unique_chunks = []
        seen_hashes = set()
        
        for chunk in chunks:
            # 计算内容哈希
            content_hash = self._calculate_content_hash(chunk['text'])
            
            if content_hash in seen_hashes:
                continue
            
            # 检查与已有块的相似性
            is_duplicate = False
            for existing_chunk in unique_chunks:
                similarity = self._calculate_similarity(chunk['text'], existing_chunk['text'])
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                seen_hashes.add(content_hash)
        
        return unique_chunks
    
    def _calculate_content_hash(self, text: str) -> str:
        """计算内容哈希"""
        # 标准化文本（移除空白、转小写）
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似性"""
        # 使用序列匹配器计算相似度
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _remove_noise(self, chunks: List[Dict]) -> List[Dict]:
        """去除噪音内容"""
        clean_chunks = []
        
        for chunk in chunks:
            text = chunk['text'].strip()
            
            # 检查是否匹配噪音模式
            is_noise = False
            for pattern in self.noise_patterns:
                if re.match(pattern, text, re.IGNORECASE):
                    is_noise = True
                    break
            
            if not is_noise:
                # 清理文本中的噪音部分
                cleaned_text = self._clean_text(text)
                if len(cleaned_text.strip()) > 30:  # 清理后仍有足够内容
                    chunk['text'] = cleaned_text
                    clean_chunks.append(chunk)
        
        return clean_chunks
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余的空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除页码等噪音
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # 移除重复的标点
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # 修复句子间距
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        return text.strip()
    
    def generate_quality_report(self, original_chunks: List[Dict], 
                              filtered_chunks: List[Dict]) -> Dict:
        """生成质量过滤报告"""
        original_count = len(original_chunks)
        filtered_count = len(filtered_chunks)
        
        # 计算统计信息
        original_word_count = sum(len(chunk['text'].split()) for chunk in original_chunks)
        filtered_word_count = sum(len(chunk['text'].split()) for chunk in filtered_chunks)
        
        report = {
            'original_chunks': original_count,
            'filtered_chunks': filtered_count,
            'retention_rate': filtered_count / original_count if original_count > 0 else 0,
            'original_word_count': original_word_count,
            'filtered_word_count': filtered_word_count,
            'word_retention_rate': filtered_word_count / original_word_count if original_word_count > 0 else 0,
            'average_chunk_quality': self._calculate_average_quality(filtered_chunks)
        }
        
        return report
    
    def _calculate_average_quality(self, chunks: List[Dict]) -> float:
        """计算平均质量分数"""
        if not chunks:
            return 0.0
        
        quality_scores = []
        for chunk in chunks:
            text = chunk['text']
            
            # 计算质量分数（0-1）
            word_count_score = min(len(text.split()) / 100, 1.0)
            diversity_score = self._calculate_character_diversity(text)
            repetition_score = 1.0 - self._calculate_repetition_ratio(text)
            
            overall_score = (word_count_score + diversity_score + repetition_score) / 3
            quality_scores.append(overall_score)
        
        return sum(quality_scores) / len(quality_scores)


# 集成测试函数
def test_complete_data_processing():
    """测试完整的数据处理流程"""
    
    # 模拟论文数据
    sample_paper = {
        'id': 'test_paper_001',
        'title': 'Advanced Machine Learning Techniques for Natural Language Processing',
        'abstract': 'This paper presents novel deep learning approaches for text classification and sentiment analysis. We propose a new neural architecture that outperforms existing methods.',
        'full_text': '''
        Abstract
        This paper presents novel deep learning approaches for text classification and sentiment analysis.
        
        1. Introduction
        Natural language processing has evolved significantly with deep learning techniques.
        We propose a new neural architecture that outperforms existing methods on benchmark datasets.
        
        2. Methodology
        Our approach combines convolutional neural networks with attention mechanisms.
        The model architecture consists of three main components: embedding layer, CNN layers, and attention module.
        
        3. Experiments
        We evaluate our method on IMDB dataset using accuracy and F1-score metrics.
        The results show significant improvement over baseline methods.
        '''
    }
    
    print("=== 完整数据处理流程测试 ===\n")
    
    # 1. 智能分块
    print("1. 智能分块处理...")
    chunker = IntelligentChunker()
    chunks = chunker.chunk_paper(sample_paper['full_text'], sample_paper['id'])
    print(f"生成了 {len(chunks)} 个文档块\n")
    
    # 2. 元数据增强
    print("2. 元数据增强...")
    enricher = MetadataEnricher()
    enriched_paper = enricher.enrich_metadata(sample_paper)
    print(f"提取的关键词: {enriched_paper['keywords'][:5]}")
    print(f"研究领域: {[d['domain'] for d in enriched_paper['research_domains'][:3]]}")
    print(f"论文类型: {enriched_paper['paper_type']['type']}\n")
    
    # 3. 质量过滤
    print("3. 质量过滤...")
    quality_filter = QualityFilter()
    filtered_chunks = quality_filter.filter_chunks(chunks)
    
    # 生成质量报告
    quality_report = quality_filter.generate_quality_report(chunks, filtered_chunks)
    print(f"质量过滤报告:")
    print(f"- 保留率: {quality_report['retention_rate']:.2%}")
    print(f"- 平均质量分数: {quality_report['average_quality']:.3f}")
    
    print("\n=== 处理完成 ===")
    
    return {
        'original_paper': sample_paper,
        'enriched_metadata': enriched_paper,
        'processed_chunks': filtered_chunks,
        'quality_report': quality_report
    }

if __name__ == "__main__":
    test_complete_data_processing()
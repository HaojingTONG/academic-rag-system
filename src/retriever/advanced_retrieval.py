# src/retriever/advanced_retrieval.py
"""
高级检索系统 - 实现混合检索、查询理解、重排序和多样性算法
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import math
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class QueryIntent:
    intent_type: str  # comparison, definition, method, recent_work, general
    confidence: float
    keywords: List[str]
    expanded_query: str

@dataclass
class RetrievalResult:
    doc_id: str
    content: str
    metadata: Dict
    vector_score: float
    bm25_score: float
    combined_score: float
    rerank_score: float = 0.0

class QueryUnderstanding:
    """查询理解模块"""
    
    def __init__(self):
        # 意图分类模式
        self.intent_patterns = {
            'comparison': [
                r'\b(compare|comparison|versus|vs|difference|differ)\b',
                r'\b(better|worse|advantage|disadvantage)\b',
                r'\b(which|what.*best|superior|inferior)\b'
            ],
            'definition': [
                r'\b(what is|define|definition|meaning|explain)\b',
                r'\b(how.*work|principle|concept)\b'
            ],
            'method': [
                r'\b(how to|method|approach|technique|algorithm)\b',
                r'\b(implement|implementation|steps|procedure)\b'
            ],
            'recent_work': [
                r'\b(recent|latest|new|current|state-of-the-art)\b',
                r'\b(progress|advance|development|trend)\b'
            ],
            'general': [r'.*']  # 默认类型
        }
        
        # AI领域同义词扩展
        self.synonyms = {
            'transformer': ['attention mechanism', 'self-attention', 'multi-head attention'],
            'cnn': ['convolutional neural network', 'convolution', 'conv net'],
            'rnn': ['recurrent neural network', 'LSTM', 'GRU', 'sequence model'],
            'gan': ['generative adversarial network', 'generator', 'discriminator'],
            'bert': ['bidirectional encoder', 'transformer encoder', 'pre-trained model'],
            'gpt': ['generative pre-trained transformer', 'autoregressive model'],
            'attention': ['attention mechanism', 'self-attention', 'cross-attention'],
            'optimization': ['gradient descent', 'backpropagation', 'training'],
            'classification': ['categorization', 'prediction', 'supervised learning']
        }
        
        # 领域特定术语扩展
        self.domain_expansions = {
            'nlp': ['natural language processing', 'text analysis', 'language model'],
            'cv': ['computer vision', 'image processing', 'visual recognition'],
            'ml': ['machine learning', 'artificial intelligence', 'AI'],
            'dl': ['deep learning', 'neural network', 'deep neural network']
        }
    
    def analyze_query(self, query: str) -> QueryIntent:
        """分析查询意图"""
        query_lower = query.lower()
        
        # 意图识别
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            intent_scores[intent] = score
        
        # 确定主要意图
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        intent_type = primary_intent[0]
        confidence = min(primary_intent[1] / len(self.intent_patterns[intent_type]), 1.0)
        
        # 提取关键词
        keywords = self.extract_keywords(query)
        
        # 查询扩展
        expanded_query = self.expand_query(query)
        
        return QueryIntent(
            intent_type=intent_type,
            confidence=confidence,
            keywords=keywords,
            expanded_query=expanded_query
        )
    
    def extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 移除停用词和标点
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def expand_query(self, query: str) -> str:
        """查询扩展"""
        expanded_terms = []
        query_lower = query.lower()
        
        # 同义词扩展
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                expanded_terms.extend(synonyms)
        
        # 领域扩展
        for domain, expansions in self.domain_expansions.items():
            if domain in query_lower:
                expanded_terms.extend(expansions)
        
        # 组合原查询和扩展词
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms[:3])}"  # 限制扩展词数量
        return query

class BM25Retriever:
    """BM25稀疏检索器"""
    
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
    
    def fit(self, documents: List[str]):
        """训练BM25模型"""
        self.documents = documents
        
        # 文档预处理和分词
        processed_docs = []
        for doc in documents:
            words = self._tokenize(doc)
            processed_docs.append(words)
            self.doc_len.append(len(words))
        
        self.avgdl = sum(self.doc_len) / len(self.doc_len)
        
        # 计算词频和IDF
        df = defaultdict(int)
        for doc in processed_docs:
            unique_words = set(doc)
            for word in unique_words:
                df[word] += 1
        
        # 计算IDF值
        num_docs = len(documents)
        for word, freq in df.items():
            self.idf[word] = math.log((num_docs - freq + 0.5) / (freq + 0.5))
        
        # 计算每个文档的词频
        self.doc_freqs = []
        for doc in processed_docs:
            freq_dict = defaultdict(int)
            for word in doc:
                freq_dict[word] += 1
            self.doc_freqs.append(freq_dict)
    
    def _tokenize(self, text: str) -> List[str]:
        """文本分词"""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25检索"""
        query_words = self._tokenize(query)
        scores = []
        
        for doc_idx in range(len(self.documents)):
            score = 0
            doc_freq = self.doc_freqs[doc_idx]
            doc_length = self.doc_len[doc_idx]
            
            for word in query_words:
                if word in doc_freq and word in self.idf:
                    tf = doc_freq[word]
                    idf = self.idf[word]
                    
                    # BM25公式
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
                    score += idf * (numerator / denominator)
            
            scores.append((doc_idx, score))
        
        # 排序并返回top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class CrossEncoderReranker:
    """Cross-encoder重排序器（简化版）"""
    
    def __init__(self):
        # 使用基于特征的简化重排序
        self.feature_weights = {
            'query_overlap': 0.3,
            'section_relevance': 0.2,
            'content_quality': 0.2,
            'length_penalty': 0.1,
            'freshness': 0.1,
            'authority': 0.1
        }
    
    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """重排序检索结果"""
        for result in results:
            features = self._extract_features(query, result)
            result.rerank_score = self._calculate_rerank_score(features)
        
        # 按重排序分数排序
        return sorted(results, key=lambda x: x.rerank_score, reverse=True)
    
    def _extract_features(self, query: str, result: RetrievalResult) -> Dict[str, float]:
        """提取重排序特征"""
        query_words = set(query.lower().split())
        content_words = set(result.content.lower().split())
        
        features = {
            # 查询词重叠率
            'query_overlap': len(query_words.intersection(content_words)) / len(query_words) if query_words else 0,
            
            # 章节相关性
            'section_relevance': self._get_section_relevance(result.metadata.get('section_type', '')),
            
            # 内容质量
            'content_quality': self._assess_content_quality(result.content, result.metadata),
            
            # 长度惩罚（太短或太长都不好）
            'length_penalty': self._calculate_length_penalty(len(result.content)),
            
            # 新鲜度（暂时固定值）
            'freshness': 0.5,
            
            # 权威性（基于引用数量等）
            'authority': 1.0 if result.metadata.get('has_citations', False) else 0.5
        }
        
        return features
    
    def _get_section_relevance(self, section_type: str) -> float:
        """章节相关性评分"""
        relevance_scores = {
            'abstract': 0.9,
            'methodology': 1.0,
            'results': 0.8,
            'introduction': 0.7,
            'conclusion': 0.8,
            'content': 0.6
        }
        return relevance_scores.get(section_type, 0.5)
    
    def _assess_content_quality(self, content: str, metadata: Dict) -> float:
        """评估内容质量"""
        quality_score = 0.5  # 基础分数
        
        # 包含公式加分
        if metadata.get('has_formulas', False):
            quality_score += 0.2
        
        # 包含代码加分
        if metadata.get('has_code', False):
            quality_score += 0.1
        
        # 包含引用加分
        if metadata.get('has_citations', False):
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def _calculate_length_penalty(self, length: int) -> float:
        """计算长度惩罚"""
        optimal_length = 500  # 理想长度
        if length < 100:
            return 0.3  # 太短
        elif length > 1500:
            return 0.6  # 太长
        else:
            # 接近理想长度的分数更高
            return 1.0 - abs(length - optimal_length) / optimal_length * 0.3
    
    def _calculate_rerank_score(self, features: Dict[str, float]) -> float:
        """计算重排序分数"""
        score = 0
        for feature, value in features.items():
            weight = self.feature_weights.get(feature, 0)
            score += weight * value
        return score

class MMRDiversifier:
    """MMR多样性算法"""
    
    def __init__(self, lambda_param=0.7):
        self.lambda_param = lambda_param  # 相关性和多样性的权衡参数
    
    def diversify(self, results: List[RetrievalResult], top_k: int = 5) -> List[RetrievalResult]:
        """使用MMR算法增加结果多样性"""
        if len(results) <= top_k:
            return results
        
        # 将结果转换为向量表示（简化版，使用TF-IDF）
        documents = [result.content for result in results]
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        try:
            doc_vectors = vectorizer.fit_transform(documents).toarray()
        except:
            # 如果向量化失败，直接返回前top_k个结果
            return results[:top_k]
        
        selected = []
        remaining = list(range(len(results)))
        
        # 选择第一个（相关性最高的）
        selected.append(0)
        remaining.remove(0)
        
        # MMR选择剩余文档
        while len(selected) < top_k and remaining:
            mmr_scores = []
            
            for idx in remaining:
                # 计算与查询的相关性（使用原始检索分数）
                relevance = results[idx].combined_score
                
                # 计算与已选文档的最大相似度
                max_similarity = 0
                for selected_idx in selected:
                    similarity = cosine_similarity(
                        doc_vectors[idx].reshape(1, -1),
                        doc_vectors[selected_idx].reshape(1, -1)
                    )[0][0]
                    max_similarity = max(max_similarity, similarity)
                
                # MMR分数
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
                mmr_scores.append((idx, mmr_score))
            
            # 选择MMR分数最高的文档
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        return [results[idx] for idx in selected]

class HybridRetriever:
    """混合检索器 - 整合所有检索组件"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.query_understanding = QueryUnderstanding()
        self.bm25_retriever = BM25Retriever()
        self.reranker = CrossEncoderReranker()
        self.diversifier = MMRDiversifier()
        self.is_fitted = False
    
    def fit(self, documents: List[str]):
        """训练混合检索器"""
        print("训练BM25检索器...")
        self.bm25_retriever.fit(documents)
        self.is_fitted = True
        print("混合检索器训练完成")
    
    def search(self, query: str, top_k: int = 10, use_reranking: bool = True, 
               use_diversity: bool = True) -> Dict:
        """混合检索"""
        
        # 1. 查询理解
        query_intent = self.query_understanding.analyze_query(query)
        print(f"查询意图: {query_intent.intent_type} (置信度: {query_intent.confidence:.2f})")
        print(f"扩展查询: {query_intent.expanded_query}")
        
        # 2. 向量检索（Dense）
        vector_results = self.vector_store.search(query_intent.expanded_query, top_k=top_k*2)
        
        # 3. BM25检索（Sparse）
        bm25_results = []
        if self.is_fitted:
            bm25_scores = self.bm25_retriever.search(query_intent.expanded_query, top_k=top_k*2)
            bm25_results = bm25_scores
        
        # 4. 结果融合
        combined_results = self._combine_results(vector_results, bm25_results, query)
        
        # 5. 重排序
        if use_reranking and len(combined_results) > 1:
            print("执行Cross-encoder重排序...")
            combined_results = self.reranker.rerank(query, combined_results)
        
        # 6. 多样性优化
        if use_diversity and len(combined_results) > top_k:
            print("应用MMR多样性算法...")
            combined_results = self.diversifier.diversify(combined_results, top_k)
        else:
            combined_results = combined_results[:top_k]
        
        return {
            'query_intent': query_intent,
            'results': combined_results,
            'retrieval_stats': {
                'vector_results': len(vector_results.get('documents', [])),
                'bm25_results': len(bm25_results),
                'combined_results': len(combined_results),
                'reranked': use_reranking,
                'diversified': use_diversity
            }
        }
    
    def _combine_results(self, vector_results: Dict, bm25_results: List, query: str) -> List[RetrievalResult]:
        """融合向量检索和BM25结果"""
        results = []
        
        # 处理向量检索结果
        vector_docs = vector_results.get('documents', [])
        vector_metas = vector_results.get('metadatas', [])
        vector_distances = vector_results.get('distances', [])
        
        for i, (doc, meta, distance) in enumerate(zip(vector_docs, vector_metas, vector_distances)):
            vector_score = 1.0 - distance  # 转换为相似度分数
            
            # 查找对应的BM25分数
            bm25_score = 0.0
            for bm25_idx, bm25_s in bm25_results:
                if bm25_idx < len(vector_docs) and i == bm25_idx:
                    bm25_score = bm25_s
                    break
            
            # 融合分数（可以调整权重）
            combined_score = 0.7 * vector_score + 0.3 * min(bm25_score / 10.0, 1.0)  # 归一化BM25分数
            
            result = RetrievalResult(
                doc_id=meta.get('chunk_id', f'doc_{i}'),
                content=doc,
                metadata=meta,
                vector_score=vector_score,
                bm25_score=bm25_score,
                combined_score=combined_score
            )
            results.append(result)
        
        # 按融合分数排序
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results

# 使用示例和测试函数
def test_hybrid_retrieval():
    """测试混合检索系统"""
    
    # 模拟文档数据
    documents = [
        "Transformer architecture uses self-attention mechanism for sequence modeling",
        "Convolutional neural networks are effective for image classification tasks",
        "BERT is a bidirectional encoder using transformer architecture",
        "Recurrent neural networks process sequential data with hidden states",
        "Generative adversarial networks consist of generator and discriminator"
    ]
    
    print("测试混合检索系统...")
    
    # 创建模拟的向量存储
    class MockVectorStore:
        def search(self, query, top_k):
            # 模拟向量检索结果
            return {
                'documents': documents[:top_k],
                'metadatas': [{'chunk_id': f'chunk_{i}', 'section_type': 'content'} for i in range(top_k)],
                'distances': [0.1 * i for i in range(top_k)]
            }
    
    # 初始化混合检索器
    mock_vs = MockVectorStore()
    hybrid_retriever = HybridRetriever(mock_vs)
    hybrid_retriever.fit(documents)
    
    # 测试查询
    test_queries = [
        "what is transformer attention mechanism",
        "compare CNN and RNN for image processing",
        "latest developments in generative models"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("-" * 50)
        
        results = hybrid_retriever.search(query, top_k=3)
        
        print(f"检索统计: {results['retrieval_stats']}")
        print(f"查询意图: {results['query_intent'].intent_type}")
        
        for i, result in enumerate(results['results']):
            print(f"\n结果 {i+1}:")
            print(f"  融合分数: {result.combined_score:.3f}")
            print(f"  向量分数: {result.vector_score:.3f}")
            print(f"  BM25分数: {result.bm25_score:.3f}")
            print(f"  重排序分数: {result.rerank_score:.3f}")
            print(f"  内容: {result.content[:100]}...")

if __name__ == "__main__":
    test_hybrid_retrieval()
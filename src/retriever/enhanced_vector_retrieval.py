"""
增强型向量检索模块
实现高级向量检索功能，包括相似度阈值过滤和智能Top-K选择
"""

from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import time

@dataclass 
class RetrievalConfig:
    """向量检索配置"""
    # Top-K参数
    default_top_k: int = 5
    max_top_k: int = 50
    min_top_k: int = 1
    
    # 相似度阈值
    similarity_threshold: float = 0.3  # 余弦相似度阈值
    strict_threshold: float = 0.5      # 严格阈值
    loose_threshold: float = 0.1       # 宽松阈值
    
    # 距离度量 
    distance_metric: str = "cosine"    # cosine, euclidean, dot_product
    
    # 智能K值调整
    enable_adaptive_k: bool = True
    relevance_drop_threshold: float = 0.15  # 相似度显著下降阈值
    
    # 结果多样性
    enable_diversity: bool = False
    diversity_threshold: float = 0.8   # 文档间相似度阈值

@dataclass
class RetrievalResult:
    """检索结果"""
    document: str
    metadata: Dict
    similarity_score: float
    distance: float
    rank: int
    is_above_threshold: bool = True

class EnhancedVectorRetriever:
    """增强型向量检索器"""
    
    def __init__(self, vector_store, config: RetrievalConfig = None):
        """
        初始化增强向量检索器
        
        Args:
            vector_store: 向量存储实例
            config: 检索配置
        """
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()
        
        print(f"🔍 初始化增强向量检索器")
        print(f"   - 默认Top-K: {self.config.default_top_k}")
        print(f"   - 相似度阈值: {self.config.similarity_threshold}")
        print(f"   - 距离度量: {self.config.distance_metric}")
        print(f"   - 自适应K值: {self.config.enable_adaptive_k}")
    
    def search(self, 
               query: str,
               top_k: Optional[int] = None,
               similarity_threshold: Optional[float] = None,
               filter_metadata: Optional[Dict] = None,
               return_scores: bool = True,
               enable_adaptive_k: Optional[bool] = None,
               enable_diversity: Optional[bool] = None) -> List[RetrievalResult]:
        """
        增强向量检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量（None使用默认值）
            similarity_threshold: 相似度阈值（None使用配置默认值）  
            filter_metadata: 元数据过滤条件
            return_scores: 是否返回分数
            enable_adaptive_k: 是否启用自适应K值调整
            enable_diversity: 是否启用结果多样性
            
        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        
        # 参数处理
        if top_k is None:
            top_k = self.config.default_top_k
        if similarity_threshold is None:
            similarity_threshold = self.config.similarity_threshold
        if enable_adaptive_k is None:
            enable_adaptive_k = self.config.enable_adaptive_k
        if enable_diversity is None:
            enable_diversity = self.config.enable_diversity
            
        # 限制top_k范围
        top_k = max(self.config.min_top_k, min(top_k, self.config.max_top_k))
        
        start_time = time.time()
        print(f"🔍 向量检索: \"{query[:50]}...\"")
        print(f"   - Top-K: {top_k}")
        print(f"   - 相似度阈值: {similarity_threshold}")
        
        # 1. 执行向量搜索（获取更多候选结果用于后续过滤）
        candidate_k = max(top_k * 2, 20) if enable_adaptive_k else top_k
        
        try:
            raw_results = self.vector_store.search(
                query=query,
                top_k=candidate_k,
                filter_metadata=filter_metadata
            )
        except Exception as e:
            print(f"❌ 向量搜索失败: {e}")
            return []
        
        # 2. 解析原始结果
        documents = raw_results.get("documents", [])
        metadatas = raw_results.get("metadatas", [])
        distances = raw_results.get("distances", [])
        
        if not documents:
            print("⚠️ 未找到任何匹配结果")
            return []
        
        # 3. 转换为RetrievalResult对象
        results = []
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # 计算相似度分数（假设使用余弦距离）
            similarity_score = self._distance_to_similarity(distance, self.config.distance_metric)
            
            # 检查是否超过阈值
            is_above_threshold = similarity_score >= similarity_threshold
            
            result = RetrievalResult(
                document=doc,
                metadata=metadata or {},
                similarity_score=similarity_score,
                distance=distance,
                rank=i + 1,
                is_above_threshold=is_above_threshold
            )
            results.append(result)
        
        # 4. 相似度阈值过滤
        filtered_results = self._apply_similarity_threshold(results, similarity_threshold)
        
        # 5. 智能Top-K调整
        if enable_adaptive_k:
            final_results = self._adaptive_top_k_selection(filtered_results, top_k)
        else:
            final_results = filtered_results[:top_k]
        
        # 6. 结果多样性处理
        if enable_diversity and len(final_results) > 1:
            final_results = self._apply_diversity_filter(final_results)
        
        # 7. 重新排序结果
        final_results = sorted(final_results, key=lambda x: x.similarity_score, reverse=True)
        for i, result in enumerate(final_results):
            result.rank = i + 1
        
        retrieval_time = time.time() - start_time
        
        # 8. 输出检索统计
        self._print_retrieval_stats(query, final_results, retrieval_time, similarity_threshold)
        
        return final_results
    
    def _distance_to_similarity(self, distance: float, metric: str) -> float:
        """将距离转换为相似度分数"""
        if metric == "cosine":
            # 余弦距离转换为余弦相似度
            return max(0.0, 1.0 - distance)
        elif metric == "euclidean":
            # 欧氏距离转换为相似度（简单的归一化方法）
            return 1.0 / (1.0 + distance)
        elif metric == "dot_product":
            # 点积已经是相似度度量
            return distance
        else:
            # 默认假设是余弦距离
            return max(0.0, 1.0 - distance)
    
    def _apply_similarity_threshold(self, 
                                   results: List[RetrievalResult], 
                                   threshold: float) -> List[RetrievalResult]:
        """应用相似度阈值过滤"""
        filtered_results = []
        below_threshold_count = 0
        
        for result in results:
            if result.similarity_score >= threshold:
                filtered_results.append(result)
            else:
                below_threshold_count += 1
                result.is_above_threshold = False
        
        if below_threshold_count > 0:
            print(f"📊 相似度过滤: 保留 {len(filtered_results)} 个结果，过滤 {below_threshold_count} 个低相似度结果")
        
        return filtered_results
    
    def _adaptive_top_k_selection(self, 
                                  results: List[RetrievalResult], 
                                  target_k: int) -> List[RetrievalResult]:
        """智能Top-K选择 - 基于相似度梯度自动调整K值"""
        if len(results) <= 1:
            return results
        
        # 计算相似度下降幅度
        similarity_drops = []
        for i in range(1, len(results)):
            drop = results[i-1].similarity_score - results[i].similarity_score
            similarity_drops.append(drop)
        
        # 寻找显著下降点
        significant_drops = []
        for i, drop in enumerate(similarity_drops):
            if drop > self.config.relevance_drop_threshold:
                significant_drops.append(i + 1)  # +1 因为索引偏移
        
        # 决定最终的K值
        final_k = target_k
        
        if significant_drops:
            # 如果有显著下降，选择第一个下降点
            suggested_k = significant_drops[0]
            if suggested_k < target_k:
                final_k = max(suggested_k, self.config.min_top_k)
                print(f"🎯 智能K值调整: {target_k} → {final_k} (检测到相似度显著下降)")
            elif suggested_k > target_k and len(results) > target_k:
                # 如果下降点在target_k之后，保持原始K值
                pass
        
        return results[:final_k]
    
    def _apply_diversity_filter(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """应用结果多样性过滤 - 移除过于相似的文档"""
        if len(results) <= 1:
            return results
        
        # 简化版多样性过滤：基于文档内容相似度
        diverse_results = [results[0]]  # 总是保留最相似的结果
        
        for result in results[1:]:
            # 检查与已选结果的相似度
            is_diverse = True
            for selected in diverse_results:
                # 简单的文本相似度检查（可以用更高级的方法）
                if self._text_similarity(result.document, selected.document) > self.config.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        if len(diverse_results) < len(results):
            print(f"🌟 多样性过滤: {len(results)} → {len(diverse_results)} 个结果")
        
        return diverse_results
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算（基于词汇重叠）"""
        if not text1 or not text2:
            return 0.0
        
        # 简单的Jaccard相似度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _print_retrieval_stats(self, 
                              query: str, 
                              results: List[RetrievalResult], 
                              retrieval_time: float,
                              threshold: float):
        """打印检索统计信息"""
        print(f"✅ 向量检索完成:")
        print(f"   - 检索时间: {retrieval_time:.3f}秒")
        print(f"   - 返回结果: {len(results)} 个")
        
        if results:
            scores = [r.similarity_score for r in results]
            print(f"   - 相似度范围: {min(scores):.3f} ~ {max(scores):.3f}")
            print(f"   - 平均相似度: {np.mean(scores):.3f}")
            
            above_threshold = sum(1 for r in results if r.is_above_threshold)
            print(f"   - 超过阈值({threshold}): {above_threshold}/{len(results)} 个")
        else:
            print(f"   - ⚠️ 没有结果超过相似度阈值 {threshold}")
    
    def search_with_multiple_thresholds(self, 
                                       query: str,
                                       thresholds: List[float] = None,
                                       top_k: int = None) -> Dict[str, List[RetrievalResult]]:
        """使用多个阈值进行检索比较"""
        if thresholds is None:
            thresholds = [
                self.config.loose_threshold,
                self.config.similarity_threshold, 
                self.config.strict_threshold
            ]
        
        if top_k is None:
            top_k = self.config.default_top_k
        
        print(f"🔍 多阈值向量检索: \"{query[:30]}...\"")
        print(f"   - 测试阈值: {thresholds}")
        
        results_by_threshold = {}
        
        for threshold in thresholds:
            threshold_key = f"threshold_{threshold:.2f}"
            results = self.search(
                query=query,
                top_k=top_k,
                similarity_threshold=threshold,
                enable_adaptive_k=False  # 禁用自适应K以便比较
            )
            results_by_threshold[threshold_key] = results
            
        # 打印比较结果
        print(f"\n📊 多阈值检索结果对比:")
        for threshold_key, results in results_by_threshold.items():
            threshold_val = float(threshold_key.split('_')[1])
            print(f"   - 阈值 {threshold_val:.2f}: {len(results)} 个结果")
        
        return results_by_threshold

    def get_retrieval_config(self) -> RetrievalConfig:
        """获取当前检索配置"""
        return self.config
    
    def update_config(self, **kwargs):
        """更新检索配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                print(f"✅ 更新配置: {key} = {value}")
            else:
                print(f"⚠️ 未知配置项: {key}")

# 工具函数
def create_retrieval_config(
    similarity_threshold: float = 0.3,
    top_k: int = 5,
    enable_adaptive_k: bool = True,
    enable_diversity: bool = False) -> RetrievalConfig:
    """创建检索配置的便捷函数"""
    
    config = RetrievalConfig()
    config.similarity_threshold = similarity_threshold
    config.default_top_k = top_k
    config.enable_adaptive_k = enable_adaptive_k
    config.enable_diversity = enable_diversity
    
    return config
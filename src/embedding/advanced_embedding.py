# src/embedding/advanced_embedding.py
"""
高级文本嵌入模块 - 支持多种嵌入模型和优化策略
"""

import os
import time
import hashlib
import pickle
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod

@dataclass
class EmbeddingConfig:
    """嵌入配置"""
    model_name: str = "all-MiniLM-L6-v2"  # 默认模型
    model_type: str = "sentence_transformers"  # 模型类型
    device: str = "auto"  # 设备选择
    batch_size: int = 32  # 批处理大小
    normalize_embeddings: bool = True  # 是否标准化向量
    cache_enabled: bool = True  # 是否启用缓存
    cache_dir: str = "data/embedding_cache"  # 缓存目录
    max_sequence_length: int = 512  # 最大序列长度

class BaseEmbeddingModel(ABC):
    """嵌入模型基类"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model_info = {}
        self._setup_device()
        self._setup_cache()
        
    def _setup_device(self):
        """设置计算设备"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device
        
        print(f"🔥 嵌入模型使用设备: {self.device}")
    
    def _setup_cache(self):
        """设置缓存"""
        if self.config.cache_enabled:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
            self.cache_enabled = True
        else:
            self.cache_enabled = False
    
    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        content = f"{self.config.model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """从缓存加载嵌入"""
        if not self.cache_enabled:
            return None
        
        cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """保存嵌入到缓存"""
        if not self.cache_enabled:
            return
        
        cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            print(f"⚠️ 缓存保存失败: {e}")
    
    @abstractmethod
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码文本"""
        pass
    
    def encode(self, texts: Union[str, List[str]], 
               show_progress: bool = True, 
               use_cache: bool = True) -> np.ndarray:
        """统一编码接口"""
        if isinstance(texts, str):
            return self.encode_single(texts)
        else:
            return self.encode_batch(texts, show_progress, use_cache)
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return self.model_info

class SentenceTransformersModel(BaseEmbeddingModel):
    """Sentence Transformers模型"""
    
    # 推荐的模型配置
    RECOMMENDED_MODELS = {
        # 英文通用模型
        "all-MiniLM-L6-v2": {
            "description": "轻量级英文模型，384维，速度快",
            "dimension": 384,
            "language": "en",
            "performance": "good",
            "speed": "fast"
        },
        "all-mpnet-base-v2": {
            "description": "高质量英文模型，768维，效果好",
            "dimension": 768,
            "language": "en", 
            "performance": "excellent",
            "speed": "medium"
        },
        
        # 多语言模型
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "description": "多语言轻量模型，384维",
            "dimension": 384,
            "language": "multilingual",
            "performance": "good",
            "speed": "fast"
        },
        "paraphrase-multilingual-mpnet-base-v2": {
            "description": "多语言高质量模型，768维",
            "dimension": 768,
            "language": "multilingual",
            "performance": "excellent", 
            "speed": "medium"
        },
        
        # 中文优化模型
        "shibing624/text2vec-base-chinese": {
            "description": "中文优化模型，768维",
            "dimension": 768,
            "language": "zh",
            "performance": "excellent",
            "speed": "medium"
        },
        
        # 学术文本专用
        "allenai/specter": {
            "description": "学术论文专用模型，768维",
            "dimension": 768,
            "language": "en",
            "performance": "excellent",
            "speed": "slow",
            "domain": "academic"
        }
    }
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._load_model()
        
    def _load_model(self):
        """加载模型"""
        print(f"📥 加载嵌入模型: {self.config.model_name}")
        
        try:
            self.model = SentenceTransformer(
                self.config.model_name, 
                device=self.device
            )
            
            # 设置最大序列长度
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = self.config.max_sequence_length
            
            # 获取模型信息
            self.model_info = {
                "model_name": self.config.model_name,
                "dimension": self.model.get_sentence_embedding_dimension(),
                "device": self.device,
                "max_sequence_length": self.config.max_sequence_length,
                "recommended_info": self.RECOMMENDED_MODELS.get(
                    self.config.model_name, {"description": "自定义模型"}
                )
            }
            
            print(f"✅ 模型加载成功")
            print(f"   - 维度: {self.model_info['dimension']}")
            print(f"   - 设备: {self.device}")
            print(f"   - 描述: {self.model_info['recommended_info']['description']}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        # 检查缓存
        cache_key = self._get_cache_key(text)
        cached_embedding = self._load_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        # 生成嵌入
        embedding = self.model.encode([text], 
                                    normalize_embeddings=self.config.normalize_embeddings,
                                    show_progress_bar=False)[0]
        
        # 保存到缓存
        self._save_to_cache(cache_key, embedding)
        
        return embedding
    
    def encode_batch(self, texts: List[str], 
                    show_progress: bool = True, 
                    use_cache: bool = True) -> np.ndarray:
        """批量编码文本"""
        if not texts:
            return np.array([])
        
        # 检查缓存
        embeddings = []
        texts_to_encode = []
        cache_keys = []
        indices_to_encode = []
        
        if use_cache and self.cache_enabled:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached_embedding = self._load_from_cache(cache_key)
                if cached_embedding is not None:
                    embeddings.append((i, cached_embedding))
                else:
                    texts_to_encode.append(text)
                    cache_keys.append(cache_key)
                    indices_to_encode.append(i)
        else:
            texts_to_encode = texts
            indices_to_encode = list(range(len(texts)))
        
        # 批量生成新的嵌入
        if texts_to_encode:
            if show_progress:
                print(f"🔢 生成 {len(texts_to_encode)} 个新嵌入向量...")
            
            # 分批处理避免内存问题
            batch_embeddings = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(texts_to_encode), batch_size):
                batch_texts = texts_to_encode[i:i + batch_size]
                batch_emb = self.model.encode(
                    batch_texts,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=show_progress and len(texts_to_encode) > batch_size,
                    batch_size=min(batch_size, len(batch_texts))
                )
                batch_embeddings.append(batch_emb)
            
            # 合并批次结果
            if batch_embeddings:
                new_embeddings = np.vstack(batch_embeddings)
                
                # 添加到结果和缓存
                for idx, embedding, cache_key in zip(indices_to_encode, new_embeddings, cache_keys):
                    embeddings.append((idx, embedding))
                    if use_cache:
                        self._save_to_cache(cache_key, embedding)
        
        # 按原始顺序排序
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])
        
        return result

class HybridEmbeddingModel(BaseEmbeddingModel):
    """混合嵌入模型 - 结合多个模型的优势"""
    
    def __init__(self, config: EmbeddingConfig, models: List[str] = None):
        super().__init__(config)
        
        # 默认模型组合
        if models is None:
            models = [
                "all-MiniLM-L6-v2",  # 快速模型
                "all-mpnet-base-v2"   # 高质量模型
            ]
        
        self.models = []
        self.weights = []  # 模型权重
        
        # 加载多个模型
        for model_name in models:
            model_config = EmbeddingConfig(
                model_name=model_name,
                device=self.device,
                batch_size=self.config.batch_size // len(models),  # 分配批次大小
                cache_enabled=self.config.cache_enabled,
                cache_dir=f"{self.config.cache_dir}/{model_name}"
            )
            model = SentenceTransformersModel(model_config)
            self.models.append(model)
            
            # 根据模型性能设置权重
            model_info = model.RECOMMENDED_MODELS.get(model_name, {})
            if model_info.get("performance") == "excellent":
                self.weights.append(0.7)
            else:
                self.weights.append(0.3)
        
        # 标准化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # 计算混合维度
        dimensions = [model.model_info["dimension"] for model in self.models]
        self.model_info = {
            "model_names": models,
            "dimension": sum(dimensions),  # 连接所有维度
            "individual_dimensions": dimensions,
            "weights": self.weights,
            "device": self.device
        }
        
        print(f"🔗 混合嵌入模型初始化完成")
        print(f"   - 模型数量: {len(self.models)}")
        print(f"   - 总维度: {self.model_info['dimension']}")
        print(f"   - 权重分配: {self.weights}")
    
    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        embeddings = []
        for model in self.models:
            emb = model.encode_single(text)
            embeddings.append(emb)
        
        # 连接所有嵌入
        return np.concatenate(embeddings)
    
    def encode_batch(self, texts: List[str], 
                    show_progress: bool = True, 
                    use_cache: bool = True) -> np.ndarray:
        """批量编码文本"""
        all_embeddings = []
        
        for i, model in enumerate(self.models):
            if show_progress:
                print(f"🔄 使用模型 {i+1}/{len(self.models)}: {model.config.model_name}")
            
            embeddings = model.encode_batch(texts, show_progress, use_cache)
            all_embeddings.append(embeddings)
        
        # 连接所有嵌入
        return np.concatenate(all_embeddings, axis=1)

class EmbeddingManager:
    """嵌入管理器 - 统一管理嵌入功能"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.stats = {
            "total_encoded": 0,
            "cache_hits": 0,
            "encoding_time": 0.0
        }
        
    def initialize(self, model_type: str = None):
        """初始化嵌入模型"""
        model_type = model_type or self.config.model_type
        
        print(f"🚀 初始化嵌入管理器...")
        print(f"   - 模型类型: {model_type}")
        print(f"   - 模型名称: {self.config.model_name}")
        
        if model_type == "sentence_transformers":
            self.model = SentenceTransformersModel(self.config)
        elif model_type == "hybrid":
            self.model = HybridEmbeddingModel(self.config)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        print(f"✅ 嵌入管理器初始化完成")
        
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """编码文本"""
        if self.model is None:
            self.initialize()
        
        start_time = time.time()
        
        # 执行编码
        embeddings = self.model.encode(texts, **kwargs)
        
        # 更新统计
        encoding_time = time.time() - start_time
        self.stats["encoding_time"] += encoding_time
        self.stats["total_encoded"] += len(texts) if isinstance(texts, list) else 1
        
        return embeddings
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self.model is None:
            return {}
        return self.model.get_model_info()
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        stats["avg_encoding_time"] = (
            stats["encoding_time"] / max(stats["total_encoded"], 1)
        )
        return stats
    
    def clear_cache(self):
        """清除缓存"""
        if self.config.cache_enabled:
            import shutil
            cache_path = Path(self.config.cache_dir)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                cache_path.mkdir(parents=True)
                print("🗑️ 嵌入缓存已清除")
    
    @staticmethod
    def get_recommended_models() -> Dict:
        """获取推荐模型列表"""
        return SentenceTransformersModel.RECOMMENDED_MODELS
    
    @staticmethod
    def benchmark_models(test_texts: List[str], 
                        models: List[str] = None) -> Dict:
        """基准测试不同模型"""
        if models is None:
            models = [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2", 
                "paraphrase-multilingual-MiniLM-L12-v2"
            ]
        
        results = {}
        
        print("🔬 开始嵌入模型基准测试...")
        
        for model_name in models:
            print(f"\n测试模型: {model_name}")
            
            try:
                config = EmbeddingConfig(
                    model_name=model_name,
                    cache_enabled=False  # 禁用缓存以获得真实性能
                )
                
                manager = EmbeddingManager(config)
                manager.initialize()
                
                # 性能测试
                start_time = time.time()
                embeddings = manager.encode(test_texts, show_progress=False)
                encoding_time = time.time() - start_time
                
                # 获取模型信息
                model_info = manager.get_model_info()
                
                results[model_name] = {
                    "success": True,
                    "encoding_time": encoding_time,
                    "avg_time_per_text": encoding_time / len(test_texts),
                    "dimension": model_info["dimension"],
                    "model_info": model_info.get("recommended_info", {}),
                    "embeddings_shape": embeddings.shape
                }
                
                print(f"✅ {model_name} 测试完成")
                print(f"   - 耗时: {encoding_time:.2f}秒")
                print(f"   - 维度: {model_info['dimension']}")
                
            except Exception as e:
                results[model_name] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"❌ {model_name} 测试失败: {e}")
        
        return results

# 使用示例和测试函数
def test_embedding_module():
    """测试嵌入模块"""
    print("🧪 测试高级嵌入模块")
    print("=" * 60)
    
    # 测试文本
    test_texts = [
        "Transformer architecture with attention mechanism",
        "Deep learning for natural language processing", 
        "Neural networks and machine learning algorithms",
        "Artificial intelligence and computer vision"
    ]
    
    # 测试不同配置
    configs = [
        {
            "model_name": "all-MiniLM-L6-v2",
            "model_type": "sentence_transformers",
            "description": "轻量级模型"
        },
        {
            "model_name": "all-mpnet-base-v2", 
            "model_type": "sentence_transformers",
            "description": "高质量模型"
        }
    ]
    
    for config in configs:
        print(f"\n🔍 测试 {config['description']}")
        print("-" * 40)
        
        embedding_config = EmbeddingConfig(
            model_name=config["model_name"],
            model_type=config["model_type"],
            cache_enabled=True
        )
        
        manager = EmbeddingManager(embedding_config)
        manager.initialize()
        
        # 编码测试
        embeddings = manager.encode(test_texts)
        model_info = manager.get_model_info()
        stats = manager.get_stats()
        
        print(f"📊 结果:")
        print(f"   - 嵌入维度: {embeddings.shape}")
        print(f"   - 模型维度: {model_info['dimension']}")
        print(f"   - 编码时间: {stats['encoding_time']:.3f}秒")
        print(f"   - 平均时间: {stats['avg_encoding_time']:.3f}秒/文本")

if __name__ == "__main__":
    test_embedding_module()
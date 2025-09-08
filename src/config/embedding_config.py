"""
嵌入模型配置统一管理
确保整个RAG系统使用一致的嵌入模型配置
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class SystemEmbeddingConfig:
    """系统级嵌入模型配置 - 确保全局一致性"""
    
    # 默认嵌入模型配置 - 所有组件必须使用相同配置
    DEFAULT_MODEL_NAME: str = "all-mpnet-base-v2"
    DEFAULT_MODEL_TYPE: str = "sentence_transformers" 
    DEFAULT_DIMENSION: int = 768
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_NORMALIZE: bool = True
    DEFAULT_CACHE_ENABLED: bool = True
    DEFAULT_CACHE_DIR: str = "data/embedding_cache"
    DEFAULT_MAX_SEQUENCE_LENGTH: int = 512
    
    @classmethod
    def get_default_config(cls):
        """获取默认嵌入配置 - 系统全局统一"""
        from src.embedding.advanced_embedding import EmbeddingConfig
        
        return EmbeddingConfig(
            model_name=cls.DEFAULT_MODEL_NAME,
            model_type=cls.DEFAULT_MODEL_TYPE,
            batch_size=cls.DEFAULT_BATCH_SIZE,
            normalize_embeddings=cls.DEFAULT_NORMALIZE,
            cache_enabled=cls.DEFAULT_CACHE_ENABLED,
            cache_dir=cls.DEFAULT_CACHE_DIR,
            max_sequence_length=cls.DEFAULT_MAX_SEQUENCE_LENGTH
        )
    
    @classmethod
    def validate_config(cls, config) -> bool:
        """验证配置是否与系统默认配置一致"""
        default_config = cls.get_default_config()
        
        # 关键参数必须一致
        critical_params = [
            'model_name',
            'model_type', 
            'normalize_embeddings',
            'max_sequence_length'
        ]
        
        for param in critical_params:
            if getattr(config, param) != getattr(default_config, param):
                print(f"⚠️ 警告: 参数 {param} 不一致!")
                print(f"   期望: {getattr(default_config, param)}")
                print(f"   实际: {getattr(config, param)}")
                return False
        
        return True
    
    @classmethod
    def get_model_info(cls) -> dict:
        """获取模型信息"""
        return {
            "model_name": cls.DEFAULT_MODEL_NAME,
            "dimension": cls.DEFAULT_DIMENSION,
            "description": "系统默认高质量嵌入模型",
            "performance": "excellent",
            "speed": "medium"
        }
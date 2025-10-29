"""Configuration loader with YAML and environment variable support."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml
import os
from loguru import logger

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # Don't override existing env vars
except ImportError:
    logger.warning("python-dotenv not installed, .env file will not be loaded")
except Exception as e:
    logger.warning(f"Failed to load .env file: {e}")


@dataclass
class SystemConfig:
    """Top-level system configuration."""
    name: str = "Academic RAG System"
    version: str = "2.0.0"
    log_level: str = "INFO"
    log_format: str = "structured"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model: str = "all-mpnet-base-v2"
    dimension: int = 768
    device: str = "auto"
    cache_dir: str = "data/embedding_cache"
    batch_size: int = 32
    max_length: int = 384
    normalize: bool = True


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    backend: str = "chromadb"
    path: str = "vector_db"
    collection_name: str = "enhanced_ai_papers_v2"
    distance_metric: str = "cosine"
    persist: bool = True


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k: int = 5
    similarity_threshold: float = 0.5
    enable_bm25: bool = True
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    enable_rerank: bool = True
    rerank_top_k: int = 10
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_mmr: bool = True
    mmr_diversity: float = 0.3
    enable_query_expansion: bool = True
    max_synonyms: int = 3


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    strategy: str = "hybrid"
    chunk_size: int = 600
    chunk_overlap: int = 100
    min_chunk_size: int = 50
    max_chunk_size: int = 1000
    preserve_tables: bool = True
    preserve_figures: bool = True
    preserve_formulas: bool = True
    preserve_code: bool = True
    preserve_citations: bool = True
    respect_section_boundaries: bool = True
    merge_short_sections: bool = True
    short_section_threshold: int = 200


@dataclass
class GenerationConfig:
    """LLM generation configuration."""
    llm_backend: str = "openai"  # "openai" or "ollama"

    # OpenAI Configuration
    openai_api_key: Optional[str] = None
    openai_api_base: str = "https://api.openai.com/v1"
    openai_organization: Optional[str] = None

    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"

    # Model Configuration
    model: str = "gpt-4o-mini"
    fallback_models: List[str] = field(default_factory=lambda: ["gpt-3.5-turbo", "gpt-4"])

    # Generation Parameters
    temperature: float = 0.1
    top_p: float = 0.9
    max_tokens: int = 2000
    timeout: int = 60

    # Prompt Settings
    system_prompt_template: str = "academic"
    include_sources: bool = True
    max_context_length: int = 4000


@dataclass
class QualityConfig:
    """Quality enhancement configuration."""
    enable_context_ranking: bool = True
    context_relevance_threshold: float = 0.6
    enable_fact_check: bool = True
    fact_check_model: str = "self"
    enable_citations: bool = True
    citation_style: str = "academic"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    enabled: bool = True
    metrics: List[str] = field(default_factory=lambda: [
        "context_relevance",
        "answer_faithfulness",
        "answer_relevance",
        "context_precision",
        "context_recall"
    ])
    llm_judge: bool = True
    llm_judge_model: str = "llama3.1:8b"
    track_baseline: bool = True
    baseline_path: str = "tests/regression/baselines"


@dataclass
class DataConfig:
    """Data paths configuration."""
    raw_papers_dir: str = "data/raw_papers"
    processed_papers: str = "data/main_system_papers.json"
    high_quality_papers: str = "data/high_quality_papers.json"
    papers_info: str = "data/papers_info.json"
    evaluation_dir: str = "data/evaluation"


@dataclass
class PDFProcessingConfig:
    """PDF processing configuration."""
    detect_sections: bool = True
    bilingual_support: bool = True
    section_types: List[str] = field(default_factory=lambda: [
        "abstract", "introduction", "related_work", "method",
        "experiment", "results", "discussion", "conclusion", "references"
    ])
    handle_dual_column: bool = True
    extract_tables: bool = True
    extract_figures: bool = True
    extract_formulas: bool = True
    preserve_short_sections: bool = True
    short_section_merge_strategy: str = "append_to_previous"


@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit: int = 100
    enable_docs: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    enable_embedding_cache: bool = True
    cache_ttl: int = 86400
    batch_size: int = 32
    max_concurrent_requests: int = 10
    max_memory_percent: int = 80
    clear_cache_on_memory_pressure: bool = True


@dataclass
class FeaturesConfig:
    """Feature flags configuration."""
    enable_web_api: bool = False
    enable_streaming: bool = False
    enable_feedback_loop: bool = False
    enable_multi_modal: bool = False
    enable_graph_rag: bool = False


@dataclass
class Config:
    """Complete system configuration."""
    system: SystemConfig = field(default_factory=SystemConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    pdf_processing: PDFProcessingConfig = field(default_factory=PDFProcessingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)


def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config dictionary.

    Environment variables should be prefixed with the section name.
    Example: EMBEDDING_MODEL overrides config['embedding']['model']
    """
    env_mappings = {
        # Embedding
        'EMBEDDING_MODEL': ('embedding', 'model'),
        'EMBEDDING_DEVICE': ('embedding', 'device'),
        'EMBEDDING_BATCH_SIZE': ('embedding', 'batch_size'),

        # Generation - OpenAI
        'OPENAI_API_KEY': ('generation', 'openai_api_key'),
        'OPENAI_API_BASE': ('generation', 'openai_api_base'),
        'OPENAI_ORGANIZATION': ('generation', 'openai_organization'),

        # Generation - Ollama
        'OLLAMA_HOST': ('generation', 'ollama_host'),

        # Generation - General
        'LLM_BACKEND': ('generation', 'llm_backend'),
        'LLM_MODEL': ('generation', 'model'),
        'OLLAMA_MODEL': ('generation', 'model'),  # Backward compatibility
        'GENERATION_TEMPERATURE': ('generation', 'temperature'),
        'GENERATION_MAX_TOKENS': ('generation', 'max_tokens'),

        # Retrieval
        'RETRIEVAL_TOP_K': ('retrieval', 'top_k'),
        'RETRIEVAL_RERANK': ('retrieval', 'enable_rerank'),

        # API
        'API_HOST': ('api', 'host'),
        'API_PORT': ('api', 'port'),
        'API_WORKERS': ('api', 'workers'),

        # Logging
        'LOG_LEVEL': ('system', 'log_level'),

        # Paths
        'DATA_DIR': ('data', 'raw_papers_dir'),
        'VECTOR_DB_PATH': ('vector_store', 'path'),
    }

    for env_var, (section, key) in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            if section in config_dict:
                # Convert types appropriately
                # IMPORTANT: Check bool BEFORE int since bool is a subclass of int in Python
                if isinstance(config_dict[section].get(key), bool):
                    value = value.lower() in ('true', '1', 'yes')
                elif isinstance(config_dict[section].get(key), int):
                    value = int(value)
                elif isinstance(config_dict[section].get(key), float):
                    value = float(value)

                config_dict[section][key] = value
                logger.debug(f"Applied env override: {env_var} -> {section}.{key} = {value}")

    return config_dict


def _nested_dict_to_config(data: Dict[str, Any]) -> Config:
    """Convert nested dictionary to Config dataclass."""
    return Config(
        system=SystemConfig(**data.get('system', {})),
        embedding=EmbeddingConfig(**data.get('embedding', {})),
        vector_store=VectorStoreConfig(**data.get('vector_store', {})),
        retrieval=RetrievalConfig(**data.get('retrieval', {})),
        chunking=ChunkingConfig(**data.get('chunking', {})),
        generation=GenerationConfig(**data.get('generation', {})),
        quality=QualityConfig(**data.get('quality', {})),
        evaluation=EvaluationConfig(**data.get('evaluation', {})),
        data=DataConfig(**data.get('data', {})),
        pdf_processing=PDFProcessingConfig(**data.get('pdf_processing', {})),
        api=APIConfig(**data.get('api', {})),
        logging=LoggingConfig(**data.get('logging', {})),
        performance=PerformanceConfig(**data.get('performance', {})),
        features=FeaturesConfig(**data.get('features', {})),
    )


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to config YAML file. If None, uses default location.

    Returns:
        Config object with all settings.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return Config()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Apply environment variable overrides
        data = _apply_env_overrides(data)

        # Convert to Config dataclass
        config_obj = _nested_dict_to_config(data)

        logger.info(f"Configuration loaded from {config_path}")
        return config_obj

    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        logger.warning("Using default configuration")
        return Config()


def validate_config(config_obj: Config) -> bool:
    """Validate configuration for common issues.

    Args:
        config_obj: Configuration to validate.

    Returns:
        True if valid, False otherwise.
    """
    issues = []

    # Check embedding dimension consistency
    if config_obj.embedding.dimension != 768 and config_obj.embedding.model == "all-mpnet-base-v2":
        issues.append("Embedding dimension mismatch: all-mpnet-base-v2 produces 768-dim vectors")

    # Check weight sum
    total_weight = config_obj.retrieval.bm25_weight + config_obj.retrieval.vector_weight
    if abs(total_weight - 1.0) > 0.01:
        issues.append(f"Retrieval weights don't sum to 1.0: {total_weight}")

    # Check chunk size
    if config_obj.chunking.chunk_overlap >= config_obj.chunking.chunk_size:
        issues.append("Chunk overlap must be less than chunk size")

    # Check paths exist
    data_dir = Path(config_obj.data.raw_papers_dir)
    if not data_dir.parent.exists():
        issues.append(f"Data directory parent doesn't exist: {data_dir.parent}")

    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False

    logger.info("Configuration validation passed")
    return True


# Global config instance
config = load_config()

# Validate on load
validate_config(config)


# Export aliases for backward compatibility
# Note: SystemConfig refers to the dataclass defined above (lines 12-17),
# not the full Config class. Use Config or config for the complete configuration.
__all__ = [
    'Config',
    'SystemConfig',
    'EmbeddingConfig',
    'VectorStoreConfig',
    'RetrievalConfig',
    'ChunkingConfig',
    'GenerationConfig',
    'QualityConfig',
    'EvaluationConfig',
    'DataConfig',
    'PDFProcessingConfig',
    'APIConfig',
    'LoggingConfig',
    'PerformanceConfig',
    'FeaturesConfig',
    'load_config',
    'validate_config',
    'config'
]

# config/settings.py
"""
Configuration Management System

Học: 
- Pydantic: Data validation và settings management
- Singleton pattern: Chỉ load config 1 lần
- Environment-based configuration
"""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== PYDANTIC MODELS ====================
# Pydantic giúp validate config và có autocomplete

class AppConfig(BaseModel):
    """Application settings"""
    name: str
    version: str
    environment:  str = Field(default="production")
    log_level: str = Field(default="INFO")
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed = ['development', 'production', 'testing']
        if v not in allowed:
            raise ValueError(f'Environment must be one of {allowed}')
        return v

class PathsConfig(BaseModel):
    """File paths"""
    data_dir: str = "data"
    pdf_file: str = "document.pdf"
    db_dir: str = "chroma_db"
    log_file: str = "rag_system.log"
    metrics_file: str = "metrics.json" # Metrics file to keep information of evaluate response
    
    @property
    def full_pdf_path(self) -> str:
        """Học: Computed property"""
        return os.path. join(self.data_dir, self.pdf_file)
    
    @property
    def full_db_path(self) -> str:
        return self.db_dir

class PDFConfig(BaseModel):
    """PDF processing settings"""
    max_file_size_mb: int = 100
    allowed_extensions: List[str] = [".pdf", ".PDF"]

class ChunkingConfig(BaseModel):
    """Text chunking settings"""
    chunk_size: int = Field(default=1000, ge=100, le=5000)  # ge=greater or equal
    chunk_overlap: int = Field(default=200, ge=0)
    separators: List[str] = ["\n\n", "\n", ". ", " ", ""]
    
    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        """Học: Cross-field validation"""
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v

class EmbeddingsConfig(BaseModel):
    """Embedding model settings"""
    model_name:  str = "all-MiniLM-L6-v2"
    device: str = "cuda"
    normalize: bool = True
    batch_size: int = 32 # Batch embedding for speed
    alternatives: Optional[Dict[str, str]] = None

class VectorDBConfig(BaseModel):
    """Vector database settings"""
    type: str = "chroma"
    collection_name:  str = "documents"
    distance_metric: str = "cosine"
    
    class HNSWConfig(BaseModel):
        space: str = "cosine"
        construction_ef: int = 100
        search_ef: int = 100
        M: int = 16
    
    hnsw:  HNSWConfig = HNSWConfig()

# MMR configuration
# MMR is a retrieval method that uses a combination of similarity and diversity to retrieve the best candidates
# Lambda multiplier is a hyperparameter that controls the balance between relevance and diversity
# Fetch k is the number of candidates to fetch
# The reason we use MMR is because it is a good balance between relevance and diversity
class MMRConfig(BaseModel):
    fetch_k: int = 20 # Fetch 20 candidates
    lambda_mult: float = Field(default=0.5, ge=0.0, le=1.0)

# Rerank configuration
# Rerank is a retrieval method that uses a re-ranking model to re-rank the candidates
# Top n is the number of candidates to return
# The reason we use rerank is because we want to re-rank the candidates to ensure that we have the best candidates
class RerankConfig(BaseModel):
    enabled: bool = False # Will implement later
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 3

# Retrieval configuration
# Retrieval is the process of retrieving the best candidates from the vector database
# The reason we use retrieval is because we want to retrieve the best candidates from the vector database
class RetrievalConfig(BaseModel):
    """Retrieval settings"""
    search_type: str = "similarity" # Changed to MMR for diversity  
    k: int = Field(default=3, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    mmr: MMRConfig = MMRConfig()
    rerank: RerankConfig = RerankConfig()

# LLM configuration
# LLM is the language model that will be used to generate the response
# The reason we use LLM is because we want to generate the response
class LLMConfig(BaseModel):
    """LLM settings"""
    provider: str = "ollama"
    model: str = "llama3"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=1)
    repeat_penalty: float = Field(default=1.1, ge=1.0)
    
    class OllamaConfig(BaseModel):
        base_url: str = "http://localhost:11434"
        timeout: int = 60
    
    ollama: OllamaConfig = OllamaConfig()

class PromptConfig(BaseModel):
    """Prompt template settings"""
    system_message: str
    rules: List[str]
    template: str = ""
    conversation_template: str | None = None # Match with config file

class ChatConfig(BaseModel):
    """Chat interface settings"""
    show_sources: bool = True
    max_query_length: int = 500
    history_enabled:  bool = False
    max_history: int = 10
    source_format: str = "detailed" # detailed | compact | none
    messages: Dict[str, str] = {} # Welcome message, exit message, processing message, error message

# Logging configuration
class ConsoleHandlerConfig(BaseModel):
    enabled: bool = True
    level: str = "INFO"

# File handler configuration
class FileHandlerConfig(BaseModel):
    enabled: bool = True
    level: str = "DEBUG"
    max_bytes: int = 10485760
    backup_count: int = 5

# Handlers configuration - Combine console and file handler
class HandlersConfig(BaseModel):
    console: ConsoleHandlerConfig = ConsoleHandlerConfig()
    file: FileHandlerConfig = FileHandlerConfig()

# Logging configuration - Combine level and format with handlers
class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: HandlersConfig = HandlersConfig()

# Metrics configuration
# Metrics is the process of tracking the performance of the system
# The reason we use metrics is because we want to track the performance of the system
class MetricsConfig(BaseModel):
    """Metrics configuration"""
    enabled: bool = True
    track: List[str] = [
        "query_count",
        "response_time",
        "retrieval_time",
        "generation_time",
        "avg_similarity_score",
    ]
    save_interval: int = 10 # Save every 10 queries
    file_path: str = "metrics.json" # Metrics file to keep information of evaluate response

# ==================== MAIN SETTINGS CLASS ====================

class Settings(BaseModel):
    """
    Main settings class that combines all configs. 
    
    Học:  Composition pattern
    """
    app: AppConfig
    paths: PathsConfig
    pdf: PDFConfig
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    vectordb: VectorDBConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    prompt: PromptConfig
    chat: ChatConfig
    logging: LoggingConfig
    metrics: MetricsConfig
    class Config:
        # Học: Pydantic config
        arbitrary_types_allowed = True
        validate_assignment = True

# ==================== CONFIG LOADER ====================

class ConfigLoader: 
    """
    Singleton config loader. 
    
    Học: Singleton pattern - Chỉ load config 1 lần
    """
    _instance:  Optional[Settings] = None
    
    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> Settings:
        """
        Load configuration from YAML file with environment overrides.
        """
        if cls._instance is not None:
            return cls._instance
        
        # Read YAML
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Environment-specific overrides
        env = os.getenv('RAG_ENVIRONMENT', config_data.get('app', {}).get('environment'))
        env_config_path = f"config/config.{env}.yaml"
        
        if os.path.exists(env_config_path):
            with open(env_config_path, 'r', encoding='utf-8') as f:
                env_config = yaml.safe_load(f)
                # Merge configs (env overrides base)
                config_data = cls._deep_merge(config_data, env_config)
        
        # Environment variable overrides (highest priority)
        cls._apply_env_overrides(config_data)
        
        # Validate and create Settings object
        cls._instance = Settings(**config_data)
        
        return cls._instance
    
    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """
        Deep merge two dictionaries.
        
        Học: Recursive dictionary merging
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @staticmethod
    def _apply_env_overrides(config_data: dict):
        """
        Apply environment variable overrides. 
        
        Học: Environment variables có priority cao nhất
        """
        # Log level override
        if log_level := os.getenv('RAG_LOG_LEVEL'):
            config_data['app']['log_level'] = log_level
            config_data['logging']['level'] = log_level
        
        # Environment override
        if env := os.getenv('RAG_ENVIRONMENT'):
            config_data['app']['environment'] = env
        
        # Add more overrides as needed
    
    @classmethod
    def reload(cls):
        """Force reload configuration"""
        cls._instance = None
        return cls.load()

# ==================== CONVENIENCE FUNCTION ====================

def get_settings() -> Settings:
    """
    Get application settings (singleton).
    
    Học:  Facade pattern - Simple interface
    """
    return ConfigLoader.load()

# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Test config loading
    settings = get_settings()
    
    print(f"App:  {settings.app.name} v{settings.app.version}")
    print(f"Environment:  {settings.app.environment}")
    print(f"Chunk size: {settings.chunking.chunk_size}")
    print(f"LLM model: {settings.llm.model}")
    print(f"Full PDF path: {settings.paths. full_pdf_path}")
    
    # Access nested config
    print(f"HNSW M value: {settings.vectordb.hnsw.M}")
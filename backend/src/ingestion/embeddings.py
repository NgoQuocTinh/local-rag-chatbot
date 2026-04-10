# src/ingestion/embeddings.py
"""
Embedding management

Lazy loading, caching
"""

from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings

from config.setting import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingManager:
    """
    Manage embedding model
    
    Singleton pattern for expensive resources
    """
    _instance: Optional['EmbeddingManager'] = None
    _embeddings: Optional[HuggingFaceEmbeddings] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.settings = get_settings()
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get embedding model (lazy load)
        
        Returns:
            HuggingFaceEmbeddings instance
        """
        if self._embeddings is not None:
            return self._embeddings
        
        try:
            config = self.settings.embeddings
            
            logger.info(f"Đang tải embedding model: {config.model_name}")
            
            self._embeddings = HuggingFaceEmbeddings(
                model_name=config.model_name,
                model_kwargs={
                    'device': config.device,
                },
                encode_kwargs={
                    'normalize_embeddings': config.normalize,
                    'batch_size': config.batch_size,  #Batch processing
                }
            )
            
            logger.info("✓ Embedding model đã sẵn sàng")
            return self._embeddings
            
        except Exception as e:
            logger.error(f"Lỗi khi tải embedding model: {str(e)}")
            raise
    
    def clear_cache(self):
        """Clear cached embeddings"""
        self._embeddings = None
        logger.info("Đã xóa embedding cache")

# Singleton instance
embedding_manager = EmbeddingManager()
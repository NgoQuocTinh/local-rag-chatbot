# ingest_v3.py
"""
Data Ingestion Pipeline - Version 3 (Config-driven)

Học: Separation of configuration from code
"""

import os
import logging
import sys
from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Import settings
from config.setting import get_settings

# ==================== SETUP ====================

sys.stdout.reconfigure(encoding="utf-8")
# Load configuration
settings = get_settings()

# Setup logging (config-driven)
logging.basicConfig(
    level=getattr(logging, settings.logging. level),
    format=settings.logging.format,
    handlers=[
        logging.FileHandler(settings.paths.log_file, encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== FUNCTIONS ====================

def validate_pdf_file() -> bool:
    """Validate PDF file using config settings"""
    file_path = settings.paths.full_pdf_path
    
    if not os.path.exists(file_path):
        logger.error(f"File không tồn tại: {file_path}")
        return False
    
    # Check extension (from config)
    if not any(file_path.endswith(ext) for ext in settings.pdf.allowed_extensions):
        logger.error(f"File không phải PDF: {file_path}")
        return False
    
    # Check file size (from config)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > settings.pdf.max_file_size_mb:
        logger. error(f"File quá lớn ({file_size_mb:.2f}MB), giới hạn {settings.pdf.max_file_size_mb}MB")
        return False
    
    logger. info(f"✓ File hợp lệ: {file_path} ({file_size_mb:.2f}MB)")
    return True

def load_pdf_documents() -> Optional[list]:
    """Load PDF documents"""
    try:
        file_path = settings.paths.full_pdf_path
        logger.info(f"Đang đọc file PDF: {file_path}")
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            logger.error("PDF không có nội dung")
            return None
        
        logger.info(f"✓ Đã tải {len(documents)} trang")
        return documents
        
    except Exception as e: 
        logger.error(f"Lỗi khi đọc PDF: {str(e)}", exc_info=True)
        return None

def split_documents(documents: list) -> Optional[list]:
    """Split documents using config settings"""
    try: 
        # Học: All parameters from config
        config = settings.chunking
        
        logger.info(
            f"Đang chia nhỏ văn bản "
            f"(size={config.chunk_size}, overlap={config.chunk_overlap})"
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=config.separators
        )
        
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            logger.error("Không tạo được chunks")
            return None
        
        logger.info(f"✓ Đã chia thành {len(chunks)} chunks")
        
        # Statistics
        lengths = [len(c.page_content) for c in chunks]
        logger.info(
            f"  - Trung bình:  {sum(lengths)/len(lengths):.0f} ký tự\n"
            f"  - Min: {min(lengths)} | Max: {max(lengths)}"
        )
        
        return chunks
        
    except Exception as e:
        logger.error(f"Lỗi khi split:  {str(e)}", exc_info=True)
        return None

def create_embeddings() -> Optional[HuggingFaceEmbeddings]:
    """Create embeddings using config settings"""
    try: 
        config = settings.embeddings
        
        logger.info(f"Đang tải embedding model: {config.model_name}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=config.model_name,
            model_kwargs={'device': config.device},
            encode_kwargs={'normalize_embeddings': config.normalize}
        )
        
        logger.info("✓ Embedding model đã sẵn sàng")
        return embeddings
        
    except Exception as e:
        logger.error(f"Lỗi khi tải embedding:  {str(e)}", exc_info=True)
        return None

def save_to_vectordb(chunks: list, embeddings: HuggingFaceEmbeddings) -> bool:
    """Save to vector DB using config settings"""
    try: 
        db_path = settings.paths.full_db_path
        
        # Check existing DB
        if os.path. exists(db_path):
            logger.warning(f"Database đã tồn tại: {db_path}")
            response = input("Ghi đè?  (y/n): ")
            if response.lower() != 'y':
                logger.info("Hủy tạo database")
                return False
            
            import shutil
            shutil.rmtree(db_path)
            logger.info("Đã xóa database cũ")
        
        logger.info(f"Đang tạo Vector Database...")
        
        # Học:  Config-driven metadata
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path,
            collection_name=settings.vectordb. collection_name,
            collection_metadata={
                "hnsw:space": settings.vectordb.hnsw.space,
                "hnsw:construction_ef": settings.vectordb. hnsw.construction_ef,
                "hnsw:M": settings.vectordb.hnsw.M
            }
        )
        
        count = vector_db._collection.count()
        logger.info(f"✓ Database đã lưu với {count} vectors")
        
        return True
        
    except Exception as e: 
        logger.error(f"Lỗi khi lưu database: {str(e)}", exc_info=True)
        return False

# ==================== MAIN ====================

def create_vector_db() -> bool:
    """Main ingestion pipeline"""
    logger.info("=" * 60)
    logger.info(f"BẮT ĐẦU INGESTION - {settings.app.name} v{settings.app.version}")
    logger.info(f"Environment: {settings.app.environment}")
    logger.info("=" * 60)
    
    # Ensure data dir exists
    Path(settings.paths.data_dir).mkdir(exist_ok=True)
    
    # Pipeline
    if not validate_pdf_file():
        return False
    
    documents = load_pdf_documents()
    if not documents:
        return False
    
    chunks = split_documents(documents)
    if not chunks:
        return False
    
    embeddings = create_embeddings()
    if not embeddings:
        return False
    
    if not save_to_vectordb(chunks, embeddings):
        return False
    
    logger.info("=" * 60)
    logger.info("✓ HOÀN TẤT!")
    logger.info("=" * 60)
    return True

if __name__ == "__main__": 
    import sys
    success = create_vector_db()
    sys.exit(0 if success else 1)
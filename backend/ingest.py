"""
Data Ingestion Pipeline - Version 4 (Multi-file support)
- Multi-file processing
- Batch operations
- Rich metadata
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config.setting import get_settings
from src.utils.logger import setup_logger
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.embeddings import embedding_manager
from langchain_chroma import Chroma

sys.stdout.reconfigure(encoding="utf-8")

logger = setup_logger(__name__)

def create_vector_db() -> bool:
    """
    Main ingestion pipeline for multiple PDFs
    
    Returns:
        True if successful, False otherwise
    """
    settings = get_settings()
    
    logger.info("=" * 70)
    logger.info(f"BẮT ĐẦU INGESTION - {settings.app.name} v{settings.app.version}")
    logger.info(f"Environment: {settings.app.environment}")
    logger.info("=" * 70)
    
    try:
        # Step 1: Process PDFs
        logger.info("\n[1/3] Xử lý PDF files...")
        processor = PDFProcessor()
        chunks = processor.process_directory()
        
        if not chunks:
            logger.error(" Không có documents để xử lý")
            return False
        
        # Show statistics
        stats = processor.get_statistics(chunks)
        logger.info(f"\n Thống kê:")
        logger.info(f"  • Số files: {stats['num_files']}")
        logger.info(f"  • Số pages: {stats['num_pages']}")
        logger.info(f"  • Số chunks: {stats['num_chunks']}")
        logger.info(f"  • Trung bình chunk: {stats['avg_chunk_size']:.0f} ký tự")
        logger.info(f"  • Files: {', '.join(stats['files'])}")
        
        # Step 2: Create embeddings
        logger.info("\n[2/3] Tạo embeddings...")
        embeddings = embedding_manager.get_embeddings()
        
        # Step 3: Save to vector DB
        logger.info("\n[3/3] Lưu vào Vector Database...")
        
        db_path = settings.paths.db_dir
        
        # Check existing DB
        if Path(db_path).exists():
            logger.warning(f"Database đã tồn tại: {db_path}")
            response = input("Bạn có muốn:\n  1. Ghi đè (overwrite)\n  2. Thêm vào (append)\n  3. Hủy\nChọn (1/2/3): ")
            
            if response == '1':
                # Overwrite
                import shutil
                shutil.rmtree(db_path)
                logger.info("Đã xóa database cũ")
            elif response == '2':
                # Append
                logger.info("Sẽ thêm documents vào database hiện có")
                vectordb = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings,
                    collection_name=settings.vectordb.collection_name
                )
                vectordb.add_documents(chunks)
                count = vectordb._collection.count()
                logger.info(f"✓ Đã thêm vào database, tổng {count} vectors")
                return True
            else:
                logger.info("Hủy tạo database")
                return False
        
        # Create new DB
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path,
            collection_name=settings.vectordb.collection_name,
            collection_metadata={
                "hnsw:space": settings.vectordb.hnsw.space,
                "hnsw:construction_ef": settings.vectordb.hnsw.construction_ef,
                "hnsw:M": settings.vectordb.hnsw.M
            }
        )
        
        count = vectordb._collection.count()
        logger.info(f"✓ Database đã lưu với {count} vectors")
        
        logger.info("\n" + "=" * 70)
        logger.info(" HOÀN TẤT! Vector Database đã sẵn sàng")
        logger.info("=" * 70)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n Đã hủy bởi người dùng")
        return False
    except Exception as e:
        logger.error(f"\n Lỗi: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # Ensure data directory exists
    settings = get_settings()
    Path(settings.paths.data_dir).mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    success = create_vector_db()
    
    sys.exit(0 if success else 1)
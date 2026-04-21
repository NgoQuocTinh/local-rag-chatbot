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
    logger.info(f"START INGESTION - {settings.app.name} v{settings.app.version}")
    logger.info(f"Environment: {settings.app.environment}")
    logger.info("=" * 70)
    
    try:
        # Step 1: Process PDFs
        logger.info("\n[1/3] Process PDF files...")
        processor = PDFProcessor()
        chunks = processor.process_directory()
        
        if not chunks:
            logger.error(" No documents to process")
            return False
        
        # Show statistics
        stats = processor.get_statistics(chunks)
        logger.info(f"\n Statistics:")
        logger.info(f"  • Number of files: {stats['num_files']}")
        logger.info(f"  • Number of pages: {stats['num_pages']}")
        logger.info(f"  • Number of chunks: {stats['num_chunks']}")
        logger.info(f"  • Average chunk size: {stats['avg_chunk_size']:.0f} characters")
        logger.info(f"  • Files: {', '.join(stats['files'])}")
        
        # Step 2: Create embeddings
        logger.info("\n[2/3] Create embeddings...")
        embeddings = embedding_manager.get_embeddings()
        
        # Step 3: Save to vector DB
        logger.info("\n[3/3] Save to Vector Database...")
        
        db_path = settings.paths.db_dir
        
        # Check existing DB
        if Path(db_path).exists():
            logger.warning(f"Database already exists: {db_path}")
            response = input("Do you want to:
  1. Overwrite
  2. Append
  3. Cancel
Choose (1/2/3): ")
            
            if response == '1':
                # Overwrite
                import shutil
                shutil.rmtree(db_path)
                logger.info("Deleted old database")
            elif response == '2':
                # Append
                logger.info("Will add documents to existing database")
                vectordb = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings,
                    collection_name=settings.vectordb.collection_name
                )
                vectordb.add_documents(chunks)
                count = vectordb._collection.count()
                logger.info(f"✓ Added to database, total {count} vectors")
                return True
            else:
                logger.info("Cancel database creation")
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
        logger.info(f"✓ Database saved with {count} vectors")
        
        logger.info("\n" + "=" * 70)
        logger.info(" COMPLETED! Vector Database is ready")
        logger.info("=" * 70)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("\n Cancelled by user")
        return False
    except Exception as e:
        logger.error(f"\n Error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    # Ensure data directory exists
    settings = get_settings()
    Path(settings.paths.data_dir).mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    success = create_vector_db()
    
    sys.exit(0 if success else 1)
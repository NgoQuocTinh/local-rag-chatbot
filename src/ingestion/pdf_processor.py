# src/ingestion/pdf_processor.py
"""
PDF processing with multi-file support
- Batch processing, metadata enrichment
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.setting import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class PDFMetadata:
    """Metadata for a PDF file"""
    filename: str
    filepath: str
    file_size_mb: float
    num_pages: int
    processed_at: str

class PDFProcessor:
    """
    Process PDF files with advanced features
    
    - Multi-file processing
    - Rich metadata
    - Error handling per file
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = self._create_text_splitter()
    
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Create text splitter from config"""
        config = self.settings.chunking
        
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=config.separators,
            add_start_index=True  # Track chunk position in original doc
        )
    
    def find_pdf_files(self, directory: Optional[str] = None) -> List[Path]:
        """
        Find all PDF files in directory
        
        Args:
            directory: Directory to search (default: from config)
        
        Returns:
            List of PDF file paths
        """
        data_dir = Path(directory or self.settings.paths.data_dir)
        
        if not data_dir.exists():
            logger.error(f"Directory không tồn tại: {data_dir}")
            return []
        
        # Find PDFs
        pdf_files = []
        for ext in self.settings.pdf.allowed_extensions:
            pdf_files.extend(data_dir.glob(f"*{ext}"))
        
        logger.info(f"Tìm thấy {len(pdf_files)} file PDF trong {data_dir}")
        
        return sorted(pdf_files)
    
    def validate_pdf(self, filepath: Path) -> bool:
        """
        Validate a single PDF file
        
        Args:
            filepath: Path to PDF file
        
        Returns:
            True if valid, False otherwise
        """
        # Check existence
        if not filepath.exists():
            logger.error(f"File không tồn tại: {filepath}")
            return False
        
        # Check extension
        if filepath.suffix.lower() not in [ext.lower() for ext in self.settings.pdf.allowed_extensions]:
            logger.error(f"File không phải PDF: {filepath}")
            return False
        
        # Check size
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        max_size = self.settings.pdf.max_file_size_mb
        
        if file_size_mb > max_size:
            logger.error(f"File quá lớn: {filepath.name} ({file_size_mb:.2f}MB > {max_size}MB)")
            return False
        
        logger.debug(f"✓ File hợp lệ: {filepath.name} ({file_size_mb:.2f}MB)")
        return True
    
    def load_single_pdf(self, filepath: Path) -> Optional[List[Document]]:
        """
        Load a single PDF file
        
        Args:
            filepath: Path to PDF file
        
        Returns:
            List of Document objects or None if failed
        """
        try:
            logger.info(f"Đang đọc: {filepath.name}")
            
            loader = PyPDFLoader(str(filepath))
            documents = loader.load()
            
            if not documents:
                logger.warning(f"PDF rỗng: {filepath.name}")
                return None
            
            # Enrich metadata
            for doc in documents:
                doc.metadata.update({
                    'filename': filepath.name,
                    'filepath': str(filepath),
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                })
            
            logger.info(f"✓ Đã đọc {len(documents)} trang từ {filepath.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Lỗi khi đọc {filepath.name}: {str(e)}")
            return None
    
    def load_multiple_pdfs(self, filepaths: List[Path]) -> List[Document]:
        """
        Load multiple PDF files
        
        Args:
            filepaths: List of PDF file paths
        
        Returns:
            Combined list of all documents
        """
        all_documents = []
        successful = 0
        failed = 0
        
        for filepath in filepaths:
            if not self.validate_pdf(filepath):
                failed += 1
                continue
            
            documents = self.load_single_pdf(filepath)
            
            if documents:
                all_documents.extend(documents)
                successful += 1
            else:
                failed += 1
        
        logger.info(
            f"Kết quả: {successful} files thành công, "
            f"{failed} files thất bại, "
            f"tổng {len(all_documents)} trang"
        )
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
        
        Returns:
            List of chunked Document objects
        """
        try:
            logger.info(f"Đang chia nhỏ {len(documents)} documents...")
            
            chunks = self.text_splitter.split_documents(documents)
            
            if not chunks:
                logger.error("Không tạo được chunks")
                return []
            
            # Statistics
            lengths = [len(c.page_content) for c in chunks]
            logger.info(
                f"✓ Đã tạo {len(chunks)} chunks\n"
                f"  - Trung bình: {sum(lengths)/len(lengths):.0f} ký tự\n"
                f"  - Min: {min(lengths)} | Max: {max(lengths)}"
            )
            
            # Học: Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_length': len(chunk.page_content),
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Lỗi khi split documents: {str(e)}")
            return []
    
    def process_directory(self, directory: Optional[str] = None) -> List[Document]:
        """
        Process all PDFs in a directory
        
        Args:
            directory: Directory path (default: from config)
        
        Returns:
            List of chunked documents ready for embedding
        """
        # Find PDFs
        pdf_files = self.find_pdf_files(directory)
        
        if not pdf_files:
            logger.error("Không tìm thấy file PDF nào")
            return []
        
        # Load PDFs
        documents = self.load_multiple_pdfs(pdf_files)
        
        if not documents:
            logger.error("Không load được document nào")
            return []
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        return chunks
    
    def get_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Get statistics about processed documents
        
        Args:
            documents: List of documents
        
        Returns:
            Statistics dictionary
        """
        if not documents:
            return {}
        
        # Count unique files
        files = set(doc.metadata.get('filename', 'unknown') for doc in documents)
        
        # Count pages
        pages = set(
            (doc.metadata.get('filename'), doc.metadata.get('page'))
            for doc in documents
        )
        
        # Calculate sizes
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chunk_size = total_chars / len(documents)
        
        return {
            'num_files': len(files),
            'num_pages': len(pages),
            'num_chunks': len(documents),
            'total_characters': total_chars,
            'avg_chunk_size': avg_chunk_size,
            'files': sorted(list(files))
        }
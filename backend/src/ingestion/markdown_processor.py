import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.setting import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MarkdownProcessor:
    """
    Xử lý các file Markdown (.md) cho hệ thống Knowledge Base kiểu Obsidian.
    Hỗ trợ trích xuất metadata và bidirectional links dạng [[Tên bài viết]].
    """
    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunking.chunk_size,
            chunk_overlap=self.settings.chunking.chunk_overlap,
            separators=self.settings.chunking.separators
        )
        
    def extract_links(self, content: str) -> List[str]:
        """
        Trích xuất tất cả các liên kết Obsidian-style [[Link]] từ nội dung.
        Hỗ trợ cả dạng có alias như [[Link|Tên hiển thị]].
        """
        pattern = r'\[\[(.*?)\]\]'
        matches = re.findall(pattern, content)
        # Chỉ lấy phần link chính, bỏ qua alias sau dấu "|"
        links = [match.split('|')[0].strip() for match in matches]
        return list(set(links)) # Xóa các link trùng lặp

    def process_file(self, file_path: Path) -> List[Document]:
        """Xử lý 1 file markdown và trả về List of Documents (chunks)"""
        try:
            content = file_path.read_text(encoding='utf-8')
            filename = file_path.name
            
            # Trích xuất các links để phục vụ Graph View sau này
            links = self.extract_links(content)
            
            metadata = {
                "source": str(file_path),
                "filename": filename,
                "type": "markdown",
                # Lưu mảng link thành chuỗi phân cách dấu phẩy để ChromaDB dễ lưu trữ
                "links": ",".join(links) if links else "" 
            }
            
            # Cắt nhỏ nội dung text
            chunks = self.text_splitter.split_text(content)
            
            documents = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk"] = i
                documents.append(Document(page_content=chunk, metadata=chunk_metadata))
                
            return documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return []
            
    def process_directory(self, directory: str = None) -> List[Document]:
        """Quét và xử lý toàn bộ file .md trong thư mục chỉ định"""
        target_dir = Path(directory) if directory else Path(self.settings.paths.data_dir)
        
        all_documents = []
        md_files = list(target_dir.glob("**/*.md"))
        
        if not md_files:
            logger.warning(f"No .md files found in {target_dir}")
            return []
            
        logger.info(f"Processing {len(md_files)} Markdown files...")
        
        for file_path in md_files:
            docs = self.process_file(file_path)
            all_documents.extend(docs)
            
        return all_documents

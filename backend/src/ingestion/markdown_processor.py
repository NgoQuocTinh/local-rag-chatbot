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
    Process Markdown files (.md) for Obsidian-style Knowledge Base system.
    Supports extracting metadata and bidirectional links in the format [[Article name]].
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
        Extract all Obsidian-style [[Link]] links from the content.
        Supports alias format like [[Link|Display Name]].
        """
        pattern = r'\[\[(.*?)\]\]'
        matches = re.findall(pattern, content)
        # Only take the main link part, ignore alias after "|"
        links = [match.split('|')[0].strip() for match in matches]
        return list(set(links)) # Remove duplicate links

    def process_file(self, file_path: Path) -> List[Document]:
        """Process a markdown file and return a List of Documents (chunks)"""
        try:
            content = file_path.read_text(encoding='utf-8')
            filename = file_path.name
            
            # Extract links to serve Graph View later
            links = self.extract_links(content)
            
            metadata = {
                "source": str(file_path),
                "filename": filename,
                "type": "markdown",
                # Save link array as comma-separated string for easy ChromaDB storage
                "links": ",".join(links) if links else "" 
            }
            
            # Split text content
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
        """Scan and process all .md files in the specified directory"""
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

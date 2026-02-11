# src/retrieval/retriever.py
"""
Advanced retrieval with MMR and filtering

Maximum Marginal Relevance, metadata filtering
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config.setting import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class AdvancedRetriever:
    """
    Advanced retrieval with multiple strategies
    
    - MMR (Maximum Marginal Relevance) - Balance relevance vs diversity
    - Metadata filtering
    - Score thresholding
    """
    
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.settings = get_settings()
    
    def retrieve(
        self,
        query: str,
        search_type: Optional[str] = None,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Retrieve documents with advanced options
        
        Args:
            query: Search query
            search_type: 'similarity' | 'mmr' (default: from config)
            k: Number of results (default: from config)
            filter: Metadata filter (e.g., {'filename': 'doc1.pdf'})
        
        Returns:
            List of retrieved documents
        """
        config = self.settings.retrieval
        search_type = search_type or config.search_type
        k = k or config.k
        
        try:
            if search_type == 'mmr':
                # MMR: Balance relevance and diversity
                docs = self.vectorstore.max_marginal_relevance_search(
                    query=query,
                    k=k,
                    fetch_k=config.mmr.fetch_k,
                    lambda_mult=config.mmr.lambda_mult,
                    filter=filter
                )
                logger.debug(f"MMR retrieval: {len(docs)} docs")
            
            elif search_type == 'similarity':
                # Standard similarity search
                docs = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter
                )
                logger.debug(f"Similarity retrieval: {len(docs)} docs")
            
            else:
                raise ValueError(f"Unknown search_type: {search_type}")
            
            return docs
            
        except Exception as e:
            logger.error(f"Lỗi khi retrieve: {str(e)}")
            return []
    
    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[tuple[Document, float]]:
        """
        Retrieve documents with similarity scores
        
        Args:
            query: Search query
            k: Number of results
            score_threshold: Minimum similarity score (0.0-1.0)
        
        Returns:
            List of (document, score) tuples
        """
        config = self.settings.retrieval
        k = k or config.k
        score_threshold = score_threshold or config.score_threshold
        
        try:
            # Get docs with scores
            docs_and_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k * 2  # Fetch more to filter
            )
            
            # Filter by threshold
            if score_threshold > 0:
                # Học: ChromaDB uses distance (lower = more similar)
                # Convert to similarity: similarity = 1 - distance
                filtered = [
                    (doc, 1 - score)
                    for doc, score in docs_and_scores
                    if (1 - score) >= score_threshold
                ]
                docs_and_scores = filtered[:k]
            else:
                docs_and_scores = [(doc, 1 - score) for doc, score in docs_and_scores[:k]]
            
            logger.debug(
                f"Retrieved {len(docs_and_scores)} docs "
                f"(threshold={score_threshold})"
            )
            
            return docs_and_scores
            
        except Exception as e:
            logger.error(f"Lỗi khi retrieve with scores: {str(e)}")
            return []
    
    def get_unique_sources(self, documents: List[Document]) -> List[str]:
        """
        Get unique source files from documents
        
        Args:
            documents: List of documents
        
        Returns:
            List of unique filenames
        """
        sources = set()
        for doc in documents:
            filename = doc.metadata.get('filename', 'unknown')
            sources.add(filename)
        return sorted(list(sources))
    
    def format_sources(
        self,
        documents: List[Document],
        format_type: str = 'detailed'
    ) -> str:
        """
        Format source citations
        
        Args:
            documents: Retrieved documents
            format_type: 'detailed' | 'compact' | 'none'
        
        Returns:
            Formatted source string
        """
        if format_type == 'none' or not documents:
            return ""
        
        if format_type == 'compact':
            # Compact: Just list unique files
            sources = self.get_unique_sources(documents)
            return f"\n\n📚 Nguồn: {', '.join(sources)}"
        
        elif format_type == 'detailed':
            # Detailed: File + page numbers
            sources_info = "\n\n📚 Nguồn tham khảo:"
            
            # Group by file
            file_pages: Dict[str, set] = {}
            for doc in documents:
                filename = doc.metadata.get('filename', 'unknown')
                page = doc.metadata.get('page', 'N/A')
                
                if filename not in file_pages:
                    file_pages[filename] = set()
                file_pages[filename].add(page)
            
            # Format
            for filename, pages in sorted(file_pages.items()):
                pages_str = ', '.join(str(p) for p in sorted(pages) if p != 'N/A')
                sources_info += f"\n  • {filename}"
                if pages_str:
                    sources_info += f" (trang {pages_str})"
            
            return sources_info
        
        return ""
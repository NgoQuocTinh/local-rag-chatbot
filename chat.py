# chat_v4.py
"""
Chat Interface - Version 4 (Conversation memory + Advanced retrieval)
- Conversation history
- MMR retrieval
- Metrics tracking
- Rich source citations
"""

import sys
import time
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config.setting import get_settings
from src.utils.logger import setup_logger
from src.utils.metrics import metrics_tracker
from src.ingestion.embeddings import embedding_manager
from src.retrieval.retriever import AdvancedRetriever
from src.chat.conversation import ConversationMemory
from src.chat.prompts import get_conversation_prompt

sys.stdout.reconfigure(encoding="utf-8")
logger = setup_logger(__name__)

class ChatBot:
    """
    Advanced RAG Chatbot with conversation memory
    
    Học: Object-oriented design for complex chatbot
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.vectorstore: Optional[Chroma] = None
        self.retriever: Optional[AdvancedRetriever] = None
        self.llm: Optional[ChatOllama] = None
        self.memory: Optional[ConversationMemory] = None
        self.chain = None
    
    def _validate_database(self) -> bool:
        """Validate database exists"""
        db_path = Path(self.settings.paths.db_dir)
        
        if not db_path.exists():
            logger.error(f"Database không tồn tại: {db_path}")
            logger.info("Chạy ingest_v4.py trước để tạo database")
            return False
        
        logger.info(f"✓ Database hợp lệ: {db_path}")
        return True
    
    def _load_vectorstore(self) -> bool:
        """Load vector database"""
        try:
            logger.info("Đang load Vector Database...")
            
            embeddings = embedding_manager.get_embeddings()
            
            self.vectorstore = Chroma(
                persist_directory=self.settings.paths.db_dir,
                embedding_function=embeddings,
                collection_name=self.settings.vectordb.collection_name
            )
            
            count = self.vectorstore._collection.count()
            logger.info(f"✓ Database loaded: {count} documents")
            
            if count == 0:
                logger.error("Database rỗng!")
                return False
            
            # Create advanced retriever
            self.retriever = AdvancedRetriever(self.vectorstore)
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi load database: {str(e)}", exc_info=True)
            return False
    
    def _initialize_llm(self) -> bool:
        """Initialize LLM"""
        try:
            config = self.settings.llm
            
            logger.info(f"Đang kết nối {config.provider}: {config.model}")
            
            self.llm = ChatOllama(
                model=config.model,
                base_url=config.ollama.base_url,
                temperature=config.temperature,
                num_predict=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                timeout=config.ollama.timeout
            )
            
            # Test connection
            self.llm.invoke("test")
            logger.info("✓ LLM sẵn sàng")
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi kết nối LLM: {str(e)}", exc_info=True)
            logger.info("Kiểm tra: ollama serve")
            return False
    
    def _create_chain(self):
        """Create RAG chain with conversation support"""
        
        def format_docs(docs):
            """Format retrieved documents"""
            if not docs:
                return "Không tìm thấy thông tin liên quan."
            
            parts = []
            for i, doc in enumerate(docs, 1):
                filename = doc.metadata.get('filename', 'unknown')
                page = doc.metadata.get('page', 'N/A')
                parts.append(
                    f"[Đoạn {i} - File: {filename}, Trang: {page}]\n"
                    f"{doc.page_content}\n"
                )
            
            return "\n".join(parts)
        
        def format_history():
            """Format conversation history"""
            if not self.settings.chat.history_enabled or not self.memory:
                return "Không có lịch sử."
            return self.memory.get_history_string()
        
        # Get prompt
        prompt = get_conversation_prompt()
        
        # Create chain
        self.chain = (
            {
                "context": lambda x: format_docs(
                    self.retriever.retrieve(x)
                ),
                "history": lambda x: format_history(),
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def initialize(self) -> bool:
        """Initialize all components"""
        logger.info("=" * 70)
        logger.info(f"KHỞI ĐỘNG {self.settings.app.name} v{self.settings.app.version}")
        logger.info(f"Environment: {self.settings.app.environment}")
        logger.info("=" * 70)
        
        # Validate database
        if not self._validate_database():
            return False
        
        # Load components
        if not self._load_vectorstore():
            return False
        
        if not self._initialize_llm():
            return False
        
        # Create chain
        try:
            self._create_chain()
            logger.info("✓ RAG Chain sẵn sàng")
        except Exception as e:
            logger.error(f"Lỗi tạo chain: {str(e)}")
            return False
        
        # Initialize conversation memory
        if self.settings.chat.history_enabled:
            self.memory = ConversationMemory()
            logger.info("✓ Conversation memory enabled")
        
        return True
    
    def process_query(self, query: str) -> Optional[str]:
        """Process user query with metrics tracking"""
        
        # Validate query
        max_len = self.settings.chat.max_query_length
        if len(query) > max_len:
            logger.warning(f"Query quá dài ({len(query)} > {max_len})")
            return f" Câu hỏi quá dài. Vui lòng giới hạn dưới {max_len} ký tự."
        
        try:
            # Start timing
            start_time = time.time()
            
            # Retrieve documents
            retrieval_start = time.time()
            docs_and_scores = self.retriever.retrieve_with_scores(query)
            retrieval_time = time.time() - retrieval_start
            
            if not docs_and_scores:
                return " Không tìm thấy thông tin liên quan trong tài liệu."
            
            docs = [doc for doc, score in docs_and_scores]
            scores = [score for doc, score in docs_and_scores]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            
            logger.info(
                f"Retrieved {len(docs)} docs "
                f"(avg similarity: {avg_score:.3f}) "
                f"in {retrieval_time:.2f}s"
            )
            
            # Generate response
            generation_start = time.time()
            response = self.chain.invoke(query)
            generation_time = time.time() - generation_start
            
            # Total time
            total_time = time.time() - start_time
            
            # Add source citations
            if self.settings.chat.show_sources:
                source_format = self.settings.chat.source_format
                sources = self.retriever.format_sources(docs, source_format)
                response += sources
            
            # Update conversation memory
            if self.memory:
                self.memory.add_user_message(query)
                self.memory.add_assistant_message(response)
            
            # Track metrics
            if self.settings.metrics.enabled:
                metrics_tracker.track_query(
                    query=query,
                    response_time=total_time,
                    retrieval_time=retrieval_time,
                    generation_time=generation_time,
                    num_retrieved_docs=len(docs),
                    avg_similarity_score=avg_score,
                    success=True
                )
            
            logger.info(
                f"Response generated in {generation_time:.2f}s "
                f"(total: {total_time:.2f}s)"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Lỗi xử lý query: {str(e)}", exc_info=True)
            
            # Track failed query
            if self.settings.metrics.enabled:
                metrics_tracker.track_query(
                    query=query,
                    response_time=0,
                    retrieval_time=0,
                    generation_time=0,
                    num_retrieved_docs=0,
                    avg_similarity_score=0,
                    success=False,
                    error=str(e)
                )
            
            return " Đã có lỗi xảy ra khi xử lý câu hỏi."
    
    def print_welcome(self):
        """Print welcome message"""
        msgs = self.settings.chat.messages
        
        print("\n" + "=" * 70)
        print(f"  {msgs.get('welcome', ' Chatbot sẵn sàng!')}")
        print("=" * 70)
        print("\n HƯỚNG DẪN:")
        print("  • Đặt câu hỏi về nội dung tài liệu")
        print("  • Gõ 'exit' hoặc 'quit' để thoát")
        print("  • Gõ 'clear' để xóa lịch sử hội thoại")
        print("  • Gõ 'stats' để xem thống kê")
        print("  • Gõ 'help' để xem hướng dẫn")
        
        if self.memory:
            print(f"  • Conversation memory: Enabled (max {self.memory.max_history} exchanges)")
        
        print("\n  CẤU HÌNH:")
        print(f"  • LLM: {self.settings.llm.model}")
        print(f"  • Retrieval: {self.settings.retrieval.search_type.upper()} (top-{self.settings.retrieval.k})")
        print(f"  • Documents in DB: {self.vectorstore._collection.count()}")
        print("=" * 70 + "\n")
    
    def run(self):
        """Main chat loop"""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n Bạn: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print(f"\n{self.settings.chat.messages.get('exit', ' Tạm biệt!')}")
                    
                    # Save metrics
                    if self.settings.metrics.enabled:
                        metrics_tracker.save()
                        print(" Metrics đã được lưu")
                    
                    break
                
                if user_input.lower() == 'clear':
                    if self.memory:
                        self.memory.clear()
                        print(" Đã xóa lịch sử hội thoại")
                    continue
                
                if user_input.lower() == 'stats':
                    if self.settings.metrics.enabled:
                        summary = metrics_tracker.get_summary()
                        print("\n THỐNG KÊ:")
                        for key, value in summary.items():
                            if isinstance(value, float):
                                print(f"  • {key}: {value:.3f}")
                            else:
                                print(f"  • {key}: {value}")
                    continue
                
                if user_input.lower() == 'help':
                    self.print_welcome()
                    continue
                
                # Process query
                print(" Bot: ", end="", flush=True)
                response = self.process_query(user_input)
                
                if response:
                    print(response)
                else:
                    print("Không thể xử lý câu hỏi. Vui lòng thử lại.")
                
            except KeyboardInterrupt:
                print(f"\n\n{self.settings.chat.messages.get('exit', ' Tạm biệt!')}")
                if self.settings.metrics.enabled:
                    metrics_tracker.save()
                break
            except Exception as e:
                logger.error(f"Lỗi không mong muốn: {str(e)}", exc_info=True)
                print(" Đã có lỗi xảy ra.")

def main():
    """Main entry point"""
    chatbot = ChatBot()
    
    if not chatbot.initialize():
        logger.error("Khởi động thất bại")
        return 1
    
    chatbot.run()
    return 0

if __name__ == "__main__":
    sys.exit(main())
# chat_v3.py
"""
Chat Interface - Version 3 (Config-driven)
"""

import os
import logging
import sys
from typing import Optional, List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core. output_parsers import StrOutputParser
from langchain_core. documents import Document

from config.setting import get_settings

# ==================== SETUP ====================
sys.stdout.reconfigure(encoding="utf-8")
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format=settings.logging.format,
    handlers=[
        logging.FileHandler(settings.paths.log_file, encoding='utf-8', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== FUNCTIONS ====================

def validate_database() -> bool:
    """Validate database exists"""
    db_path = settings. paths.full_db_path
    
    if not os.path.exists(db_path):
        logger.error(f"Database không tồn tại: {db_path}")
        logger.info("Chạy ingest_v3.py trước")
        return False
    
    logger.info(f"✓ Database hợp lệ: {db_path}")
    return True

def validate_query(query: str) -> bool:
    """Validate user query"""
    if not query or not query.strip():
        return False
    
    max_len = settings.chat.max_query_length
    if len(query) > max_len:
        logger.warning(f"Query quá dài ({len(query)} chars), giới hạn {max_len}")
        return False
    
    return True

def load_vectordb() -> Optional[Chroma]:
    """Load vector database"""
    try:
        logger.info("Đang load Vector Database...")
        
        config = settings.embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=config.model_name,
            model_kwargs={'device': config.device},
            encode_kwargs={'normalize_embeddings':  config.normalize}
        )
        
        vector_db = Chroma(
            persist_directory=settings.paths.full_db_path,
            embedding_function=embeddings,
            collection_name=settings.vectordb.collection_name
        )
        
        count = vector_db._collection.count()
        logger.info(f"✓ Database loaded:  {count} documents")
        
        return vector_db if count > 0 else None
        
    except Exception as e: 
        logger.error(f"Lỗi load database: {str(e)}", exc_info=True)
        return None

def initialize_llm() -> Optional[ChatOllama]:
    """Initialize LLM"""
    try:
        config = settings.llm
        
        logger.info(f"Đang kết nối {config.provider}:  {config.model}")
        
        llm = ChatOllama(
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
        llm.invoke("test")
        logger.info("✓ LLM sẵn sàng")
        
        return llm
        
    except Exception as e: 
        logger.error(f"Lỗi kết nối LLM: {str(e)}", exc_info=True)
        logger.info("Kiểm tra:  ollama serve")
        return None

def format_docs(docs: List[Document]) -> str:
    """Format documents for context"""
    if not docs: 
        return "Không tìm thấy thông tin liên quan."
    
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get('page', 'N/A')
        source = doc.metadata.get('source', 'N/A')
        parts.append(f"[Đoạn {i} - Trang {page}]\n{doc.page_content}\n")
    
    return "\n".join(parts)

def create_rag_chain(vector_db: Chroma, llm: ChatOllama):
    """Create RAG chain"""
    # Retriever with config
    retrieval_config = settings.retrieval
    
    retriever = vector_db.as_retriever(
        search_type=retrieval_config.search_type,
        search_kwargs={
            "k": retrieval_config.k,
        }
    )
    
    # Prompt from config
    prompt_config = settings.prompt
    
    # Build rules string
    rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(prompt_config.rules)])
    
    template = prompt_config.template. format(
        system_message=prompt_config.system_message,
        rules=rules_text,
        context="{context}",
        question="{question}"
    )
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

def process_query(query: str, rag_chain, retriever) -> Optional[str]:
    """Process user query"""
    if not validate_query(query):
        return None
    
    try:
        logger.info(f"Query: {query}")
        
        # Retrieve
        docs = retriever.invoke(query)
        logger.info(f"Retrieved {len(docs)} documents")
        
        # Generate
        response = rag_chain. invoke(query)
        
        # Add sources if enabled
        if settings.chat.show_sources and docs:
            sources = "\n\n📚 Nguồn:"
            for doc in docs:
                page = doc.metadata.get('page', 'N/A')
                sources += f"\n  - Trang {page}"
            response += sources
        
        return response
        
    except Exception as e:
        logger.error(f"Lỗi xử lý query: {str(e)}", exc_info=True)
        return settings.chat.messages.get('error', 'Đã có lỗi xảy ra')

# ==================== MAIN ====================

def start_chat():
    """Main chat loop"""
    logger.info("=" * 60)
    logger.info(f"KHỞI ĐỘNG {settings.app.name} v{settings.app.version}")
    logger.info(f"Environment: {settings.app.environment}")
    logger.info("=" * 60)
    
    # Validate & load
    if not validate_database():
        return
    
    vector_db = load_vectordb()
    if not vector_db: 
        return
    
    llm = initialize_llm()
    if not llm:
        return
    
    try:
        rag_chain, retriever = create_rag_chain(vector_db, llm)
        logger.info("✓ RAG Chain sẵn sàng")
    except Exception as e:
        logger. error(f"Lỗi tạo RAG chain: {str(e)}")
        return
    
    # Chat loop with config messages
    msgs = settings.chat.messages
    print(f"\n{msgs. get('welcome', '🤖 Bot sẵn sàng!')}")
    print("=" * 60)
    print("Commands:  'exit', 'quit', 'help'")
    print("=" * 60 + "\n")
    
    while True:
        try: 
            query = input("\n🧑 Bạn:  ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print(msgs.get('exit', '👋 Tạm biệt! '))
                break
            
            if query.lower() == 'help':
                print("\n📖 HƯỚNG DẪN:")
                print("  - Đặt câu hỏi về tài liệu")
                print("  - 'exit' để thoát")
                print(f"  - Model: {settings.llm.model}")
                print(f"  - Retrieval: Top-{settings.retrieval.k}")
                continue
            
            if not query: 
                continue
            
            print("🤖 Bot:  ", end="", flush=True)
            response = process_query(query, rag_chain, retriever)
            
            if response:
                print(response)
            else:
                print("Không xử lý được.  Thử lại.")
                
        except KeyboardInterrupt:
            print(f"\n\n{msgs.get('exit', '👋 Tạm biệt!')}")
            break
        except Exception as e:
            logger.error(f"Lỗi:  {str(e)}", exc_info=True)
            print(msgs.get('error', 'Lỗi xảy ra'))

if __name__ == "__main__":
    start_chat()
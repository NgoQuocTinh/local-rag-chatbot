from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# CẤU HÌNH
DB_PATH = "chroma_db"

def start_chat():
    print("--- Đang khởi động Bot... ---")
    
    # 1. Load lại Database đã lưu
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    # Biến DB thành công cụ tìm kiếm (Retriever)
    # k=3 nghĩa là tìm 3 đoạn văn bản liên quan nhất
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # 2. Khởi tạo LLM (Ollama chạy Llama 3)
    llm = ChatOllama(model="llama3")

    # 3. Tạo Prompt Template (Hướng dẫn AI cách trả lời)
    # Đây là kỹ thuật Prompt Engineering: Ép AI chỉ trả lời dựa trên Context
    template = """Bạn là trợ lý AI thông minh. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp dưới đây.
    Nếu thông tin không có trong văn bản, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu".
    
    Thông tin ngữ cảnh (Context):
    {context}
    
    Câu hỏi của người dùng: {question}
    
    Trả lời bằng tiếng Việt chi tiết:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Tạo Chain (Quy trình xử lý: Tìm kiếm -> Ghép Prompt -> Gửi cho LLM)
    # Dùng LCEL (LangChain Expression Language) - Chuẩn code hiện đại
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("--- Bot đã sẵn sàng! Gõ 'exit' để thoát ---")
    
    # 5. Vòng lặp Chat
    while True:
        query = input("\nBạn: ")
        if query.lower() == "exit":
            break
        
        # Gọi Chain để xử lý
        print("Bot đang suy nghĩ...", end="\r")
        try:
            response = rag_chain.invoke(query)
            print(f"\nBot: {response}")
        except Exception as e:
            print(f"\nLỗi: {e}")

if __name__ == "__main__":
    start_chat()
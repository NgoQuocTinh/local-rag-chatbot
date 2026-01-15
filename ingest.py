import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# CẤU HÌNH
DATA_PATH = "data/document.pdf"  # Đường dẫn file PDF của bạn
DB_PATH = "chroma_db"            # Nơi lưu Vector Database

def create_vector_db():
    # 1. Kiểm tra file tồn tại
    if not os.path.exists(DATA_PATH):
        print(f"LỖI: Không tìm thấy file {DATA_PATH}. Hãy tạo folder 'data' và bỏ file PDF vào.")
        return

    print("--- 1. Đang đọc file PDF... ---")
    loader = PyPDFLoader(DATA_PATH)
    documents = loader.load()
    print(f"   > Đã tải {len(documents)} trang tài liệu.")

    # 2. Chia nhỏ văn bản (Chunking)
    # Tại sao? Vì LLM có giới hạn bộ nhớ, và để tìm kiếm chính xác hơn.
    print("--- 2. Đang chia nhỏ văn bản (Chunking)... ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Mỗi đoạn khoảng 1000 ký tự
        chunk_overlap=200   # Gối đầu 200 ký tự để không mất ngữ cảnh ở vết cắt
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   > Đã chia thành {len(chunks)} đoạn nhỏ.")

    # 3. Khởi tạo mô hình Embedding (Biến chữ thành số)
    # Sử dụng model local miễn phí 'all-MiniLM-L6-v2' rất nhẹ và nhanh
    print("--- 3. Đang tạo Vector Database... ---")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Lưu vào ChromaDB (Lưu xuống ổ cứng tại folder DB_PATH)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print(f"--- THÀNH CÔNG! Database đã được lưu tại folder '{DB_PATH}' ---")

if __name__ == "__main__":
    create_vector_db()
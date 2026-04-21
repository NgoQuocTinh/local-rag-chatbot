import os
import glob
import re

dir_path = r'c:\NQT\local-rag-chatbot\backend'

replacements = {
    r'Pydantic: Data validation và settings management': 'Pydantic: Data validation and settings management',
    r'Singleton pattern: Chỉ load config 1 lần': 'Singleton pattern: Only load config once',
    r'Learn: Environment variables có priority cao nhất': 'Learn: Environment variables have the highest priority',
    r'Trung bình: (.*?) ký tự': r'Average: \1 characters',
    r'ký tự': 'characters',
    r' Không có lịch sử\.': ' No history.',
    r'KHỞI ĐỘNG': 'STARTING',
    r'RAG Chain sẵn sàng': 'RAG Chain ready',
    r'Lỗi tạo chain:': 'Error creating chain:',
    r'Query quá dài': 'Query too long',
    r'Câu hỏi quá dài\. Vui lòng giới hạn dưới (.*?) ký tự\.': r'Question too long. Please limit to \1 characters.',
    r'Không tìm thấy thông tin liên quan trong tài liệu\.': 'No relevant information found in the documents.',
    r'Lỗi xử lý query:': 'Error processing query:',
    r'Đã có lỗi xảy ra khi xử lý câu hỏi\.': 'An error occurred while processing the question.',
    r'Chatbot sẵn sàng!': 'Chatbot ready!',
    r'HƯỚNG DẪN:': 'INSTRUCTIONS:',
    r'Đặt câu hỏi về nội dung tài liệu': 'Ask questions about the document content',
    r'Gõ \'exit\' hoặc \'quit\' để thoát': 'Type \'exit\' or \'quit\' to exit',
    r'Gõ \'clear\' để xóa lịch sử hội thoại': 'Type \'clear\' to clear conversation history',
    r'Gõ \'stats\' để xem thống kê': 'Type \'stats\' to view statistics',
    r'Gõ \'help\' để xem hướng dẫn': 'Type \'help\' to view instructions',
    r'CẤU HÌNH:': 'CONFIGURATION:',
    r'THỐNG KÊ:': 'STATISTICS:',
    r'Bạn có muốn:\\n  1. Overwrite\\n  2. Append\\n  3. Cancel\\nChọn \(1/2/3\):': 'Do you want to:\\n  1. Overwrite\\n  2. Append\\n  3. Cancel\\nChoose (1/2/3):',
    r'Đã tạo database mới\. Tổng vectors:': 'Created new database. Total vectors:',
    r'Upload PDF vào data_dir': 'Upload PDF to data_dir',
    r'Đã lưu (.*?) file vào': r'Saved \1 files to',
    r'Chưa có PDF trong': 'No PDFs found in',
    r'Tìm thấy \*\*(.*?)\*\* file:': r'Found **\1** files:',
    r'Ingest vào ChromaDB': 'Ingest to ChromaDB',
    r'Chế độ': 'Mode',
    r'Chạy ingest': 'Run ingest',
    r'Đang ingest\.\.\.': 'Ingesting...',
    r'Chưa có database\. Hãy chạy Ingest trước\.': 'No database found. Please run Ingest first.',
    r'Hiển thị ngữ cảnh đã retrieve \(để kiểm chứng\)': 'Show retrieved context (for verification)',
    r'Nhập câu hỏi của bạn\.\.\.': 'Enter your question...',
    r'Đang trả lời\.\.\.': 'Answering...',
    r'Không có lịch sử\.': 'No history.',
}

for root, dirs, files in os.walk(dir_path):
    if 'node_modules' in root or '.venv' in root or '__pycache__' in root:
        continue
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            for vi, en in replacements.items():
                content = re.sub(vi, en, content)
            
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f'Replaced in: {filepath}')

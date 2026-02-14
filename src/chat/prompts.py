# src/chat/prompts.py
"""
Prompt templates for chat

Học: Centralized prompt management
"""

from typing import Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config.setting import get_settings

def get_rag_prompt() -> ChatPromptTemplate:
    """
    Get RAG prompt template (without conversation history)
    
    Returns:
        ChatPromptTemplate
    """
    settings = get_settings()
    config = settings.prompt
    
    # Build rules string
    rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(config.rules)])
    
    template = f"""{config.system_message}

    QUY TẮC:
    {rules_text}

    NGỮ CẢNH TÀI LIỆU:
    {{context}}

    CÂU HỎI: {{question}}

    TRẢ LỜI:"""
    
    return ChatPromptTemplate.from_template(template)

def get_conversation_prompt() -> ChatPromptTemplate:
    """
    Get conversation-aware RAG prompt
    
    Using conversation history in prompt
    
    Returns:
        ChatPromptTemplate with history placeholder
    """
    settings = get_settings()
    config = settings.prompt
    
    # Build rules string
    rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(config.rules)])
    
    template = f"""{config.system_message}

    QUY TẮC:
    {rules_text}

    LỊCH SỬ HỘI THOẠI:
    {{history}}

    NGỮ CẢNH TÀI LIỆU:
    {{context}}

    YÊU CẦU:
    - Trả lời như một gia sư, giải thích rõ ràng, từng bước.
    - Mỗi kết luận/định nghĩa chính phải dẫn nguồn: tên file, chương/mục, trang.
    - Nếu suy luận thêm ngoài tài liệu, phải đánh dấu rõ là "suy luận".

    CÂU HỎI MỚI: {{question}}

    TRẢ LỜI:"""
    
    return ChatPromptTemplate.from_template(template)

def get_followup_prompt() -> ChatPromptTemplate:
    """
    Get prompt for follow-up questions
    
    Học: Context-aware follow-up handling
    """
    template = """Dựa trên cuộc hội thoại trước đó và câu hỏi mới, hãy trả lời một cách nhất quán.

    LỊCH SỬ:
    {history}

    NGỮ CẢNH BỔ SUNG:
    {context}

    CÂU HỎI TIẾP THEO: {question}

    TRẢ LỜI (giữ tính nhất quán với câu trả lời trước):"""
    
    return ChatPromptTemplate.from_template(template)
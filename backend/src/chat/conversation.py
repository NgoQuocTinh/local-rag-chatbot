"""
Conversation memory management

Học: Stateful conversation in RAG
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from config.setting import get_settings

@dataclass
class Message:
    """A single message in conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str

class ConversationMemory:
    """
    Manage conversation history
    
    Học: Sliding window memory (keep last N exchanges)
    """
    
    def __init__(self, max_history: Optional[int] = None):
        settings = get_settings()
        self.max_history = max_history or settings.chat.max_history
        self.messages: List[Message] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to history"""
        from datetime import datetime
        
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        
        self. messages.append(message)
        
        # Keep only last N exchanges (2 messages = 1 exchange)
        max_messages = self.max_history * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
    
    def add_user_message(self, content: str):
        """Add user message"""
        self.add_message('user', content)
    
    def add_assistant_message(self, content: str):
        """Add assistant message"""
        self. add_message('assistant', content)
    
    def get_history_string(self) -> str:
        """
        Format history as string for prompt
        
        Returns:
            Formatted history string
        """
        if not self.messages:
            return "Không có lịch sử hội thoại trước đó."
        
        history_parts = []
        for msg in self.messages:
            role_label = "Người dùng" if msg.role == 'user' else "Trợ lý"
            history_parts.append(f"{role_label}: {msg.content}")
        
        return "\n".join(history_parts)
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get messages as list of dicts (for LangChain)"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
    
    def __len__(self) -> int:
        """Number of messages"""
        return len(self.messages)
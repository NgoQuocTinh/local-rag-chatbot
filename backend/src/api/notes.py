from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import base64
import os
import time

from config.setting import get_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Tạo Router để gộp chung nhóm API về Ghi chú
router = APIRouter(prefix="/api/notes", tags=["Notes"])
settings = get_settings()

# Đảm bảo thư mục lưu trữ data luôn tồn tại
DATA_DIR = Path(settings.paths.data_dir).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---- Khai báo Models (Kiểu dữ liệu nhận từ Client) ----
class NoteCreate(BaseModel):
    title: str
    content: str

class NoteUpdate(BaseModel):
    content: str

# ---- Hàm Encode / Decode / Sanitize Security ----
def encode_id(filename: str) -> str:
    """Encode filename to Base64 ID for URL safety (Note: This is obfuscation, not encryption)"""
    return base64.urlsafe_b64encode(filename.encode("utf-8")).decode("utf-8")

def decode_id(note_id: str) -> str:
    """Decode Base64 ID back to filename"""
    try:
        return base64.urlsafe_b64decode(note_id.encode("utf-8")).decode("utf-8")
    except Exception as e:
        logger.error(f"Error decoding file ID: {e}")
        raise HTTPException(status_code=400, detail="Invalid Note ID format")

def get_safe_file_path(filename: str) -> Path:
    """
    Core Security Logic: Prevent Directory Traversal Attacks.
    Resolves the absolute path and ensures it strictly stays within DATA_DIR.
    """
    if not filename.endswith('.md'):
        filename += '.md'
    
    # 1. Loại bỏ các ký tự chéo / và \ có thể dùng để nhảy thư mục
    clean_filename = filename.replace("/", "").replace("\\", "")
    
    # 2. Xóa các tiền tố "../" có thể còn sót lại
    while ".." in clean_filename:
        clean_filename = clean_filename.replace("..", "")
    
    # 3. Tính toán đường dẫn tuyệt đối (Absolute resolve)
    file_path = (DATA_DIR / clean_filename).resolve()
    
    # 4. Chốt chặn cuối cùng: Đường dẫn thực tế bắt buộc phải nằm bên trong DATA_DIR
    try:
        # Python 3.9+ hỗ trợ hàm này, nếu báo lỗi thì nó nằm ngoài phạm vi
        if not file_path.is_relative_to(DATA_DIR.resolve()):
            raise ValueError()
    except (ValueError, AttributeError):
        # Fallback bằng chuỗi nếu lỗi version
        if not str(file_path).startswith(str(DATA_DIR.resolve())):
            logger.error(f"Directory Traversal blocked for path: {file_path}")
            raise HTTPException(status_code=403, detail="Unauthorized file access detected!")
        
    return file_path

# ---- API Endpoints ----

@router.get("/")
def get_all_notes():
    """Lấy danh sách toàn bộ các file Markdown hiện có (Sidebar)"""
    notes = []
    try:
        for file_path in DATA_DIR.glob("**/*.md"):
            # Encode tên file thành ID Base64
            safe_id = encode_id(file_path.name)
            notes.append({
                "id": safe_id,
                "title": file_path.stem,  # Tên file không có đuôi .md
                "updated_at": file_path.stat().st_mtime
            })
        # Sắp xếp bài mới cập nhật lên đầu
        notes.sort(key=lambda x: x["updated_at"], reverse=True)
        return {"notes": notes}
    except Exception as e:
        logger.error(f"Error reading notes list: {e}")
        raise HTTPException(status_code=500, detail="Could not load notes list")

@router.get("/{note_id}")
def get_note_content(note_id: str):
    """Đọc nội dung của một bài viết thông qua Base64 ID"""
    filename = decode_id(note_id)
    if not filename.endswith('.md'):
        filename += '.md'
        
    file_path = DATA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Note not found")
        
    return {
        "id": note_id,
        "title": file_path.stem,
        "content": file_path.read_text(encoding='utf-8')
    }

@router.post("/")
def create_note(note: NoteCreate):
    """Tạo một file Markdown mới"""
    # Làm gọn tên file (xóa khoảng trắng thừa) & kiểm tra bảo mật
    safe_title = note.title.strip().replace("/", "_").replace("\\", "_")
    filename = f"{safe_title}.md"
    file_path = DATA_DIR / filename
    
    if file_path.exists():
        raise HTTPException(status_code=400, detail="Note title already exists! Please choose another name.")
        
    try:
        file_path.write_text(note.content, encoding='utf-8')
        return {"message": "Note created successfully", "id": encode_id(filename)}
    except Exception as e:
        logger.error(f"Error creating note {filename}: {e}")
        raise HTTPException(status_code=500, detail="Error saving file")

@router.put("/{note_id}")
def update_note(note_id: str, note: NoteUpdate):
    """Cập nhật nội dung một bài viết đang có bằng ID"""
    filename = decode_id(note_id)
    if not filename.endswith('.md'):
        filename += '.md'
        
    file_path = DATA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Note not found")
        
    try:
        file_path.write_text(note.content, encoding='utf-8')
        return {"message": "Note updated successfully", "id": note_id}
    except Exception as e:
        logger.error(f"Error updating note {filename}: {e}")
        raise HTTPException(status_code=500, detail="Error updating file")

@router.delete("/{note_id}")
def delete_note(note_id: str):
    """Xóa một bài viết thông qua ID"""
    filename = decode_id(note_id)
    if not filename.endswith('.md'):
        filename += '.md'
        
    file_path = DATA_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Note not found")
        
    try:
        file_path.unlink()  # Xóa file
        return {"message": "Note deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting note {filename}: {e}")
        raise HTTPException(status_code=500, detail="Could not delete this file")

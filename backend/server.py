import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.setting import get_settings

# Khởi tạo Settings
settings = get_settings()

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title=f"{settings.app.name} API",
    version=settings.app.version,
    description="Knowledge Base & RAG Backend API"
)

# Cấu hình CORS để Frontend (Next.js) có thể gọi được API
origins = [
    "http://localhost:3000",   # Mặc định Next.js
    "http://localhost:5173",   # Mặc định Vite/SvelteKit
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", tags=["System"])
def health_check():
    """Kiểm tra trạng thái hệ thống"""
    return {
        "status": "healthy",
        "app_name": settings.app.name,
        "version": settings.app.version,
        "environment": settings.app.environment
    }

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True # Tự động restart khi sửa code (cho môi trường dev)
    )

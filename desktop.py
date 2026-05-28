import os
import sys
import threading
import time
import urllib.request
import uvicorn
import webview
from fastapi.staticfiles import StaticFiles

def get_base_path():
    """Lấy đường dẫn gốc của thư mục, tương thích với PyInstaller khi đóng gói file exe"""
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

base_path = get_base_path()

# Quan trọng: Chuyển thư mục làm việc (Current Working Directory) vào "backend"
# Để các đường dẫn tương đối như "data/documents" hay "chroma_db" hoạt động chính xác!
backend_dir = os.path.join(base_path, "backend")
if os.path.exists(backend_dir):
    os.chdir(backend_dir)

# Import FastAPI app từ backend/server.py
# Nạp cả thư mục gốc và thư mục backend vào sys.path để Python nhận diện được package
sys.path.insert(0, base_path)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from backend.server import app

frontend_out = os.path.join(base_path, 'frontend', 'out')

# Mount giao diện Frontend (file tĩnh đã được Next.js build ra) vào FastAPI
if os.path.exists(frontend_out):
    app.mount("/", StaticFiles(directory=frontend_out, html=True), name="frontend")
else:
    print(f"UI not found: {frontend_out}")
    print("Please run 'npm run build' in the frontend directory first!")

def run_server():
    """Khởi chạy server FastAPI ẩn trong background"""
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

# Giao diện HTML Loading tuỳ chỉnh (sẽ hiển thị trong lúc chờ AI models load)
loading_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { background-color: #0f172a; color: #f8fafc; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; font-family: system-ui, -apple-system, sans-serif; }
        .spinner { border: 4px solid #1e293b; border-top: 4px solid #3b82f6; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin-bottom: 20px; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        p { font-size: 1.1rem; }
        .sub { color: #94a3b8; font-size: 0.9rem; margin-top: 10px; max-width: 400px; text-align: center; line-height: 1.5; }
    </style>
</head>
<body>
    <div class="spinner"></div>
    <p>Personal Knowledge Assistant is loading...</p>
    <div class="sub">The system is loading AI (LLMs & Embeddings) into RAM. This process may take a few minutes on the first run. Please wait...</div>
</body>
</html>
"""

def check_server_and_redirect(window):
    """Liên tục kiểm tra xem Web Server đã sẵn sàng chưa, sau đó chuyển hướng webview"""
    url = "http://127.0.0.1:8000"
    while True:
        try:
            # Gửi 1 cú ping nhỏ đến server
            res = urllib.request.urlopen(url, timeout=1)
            if res.getcode() == 200:
                break
        except Exception:
            time.sleep(1) # Đợi 1 giây rồi thử lại
    
    # Khi server đã sống, lập tức chuyển cửa sổ sang giao diện Next.js
    window.load_url(url)

if __name__ == "__main__":
    # 1. Chạy tiến trình ảo (Thread) cho Server Backend (Python + FastAPI)
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    
    # 2. Mở một cửa sổ Desktop native ảo (WebView2 Windows / WebKit Mac)
    # Khởi tạo với màn hình Loading tĩnh tự vẽ siêu nhẹ
    window = webview.create_window(
        title="Personal Knowledge Assistant", 
        html=loading_html,
        width=1200,
        height=800,
        min_size=(800, 600)
    )
    
    # Khởi động giao diện và gọi hàm chạy ngầm `check_server_and_redirect`
    webview.start(func=check_server_and_redirect, args=(window,))

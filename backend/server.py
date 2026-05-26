import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from config.setting import get_settings

from src.api.notes import router as notes_router
from src.api.graph import router as graph_router
from src.api.ai import router as ai_router

# Initialize Settings
settings = get_settings()

# Initialize FastAPI application
app = FastAPI(
    title=f"{settings.app.name} API",
    version=settings.app.version,
    description="Knowledge Base & RAG Backend API"
)

# Add routes for managing Notes, Graph, and AI
app.include_router(notes_router)
app.include_router(graph_router)
app.include_router(ai_router)

# Configure CORS so Frontend (Next.js) can call API
origins = [
    "http://localhost:3000",   # Default Next.js
    "http://localhost:5173",   # Default Vite/SvelteKit
    "*",                       # Allow all origins for Production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health", tags=["System"])
def health_check():
    """Check system status"""
    return {
        "status": "healthy",
        "app_name": settings.app.name,
        "version": settings.app.version,
        "environment": settings.app.environment
    }

# --- DISABLED PRELOAD FOR RENDER DEPLOYMENT ---
@app.on_event("startup")
async def startup_event():
    """Preload AI models into RAM/VRAM (Warm-up) to reduce latency on first request.

    This runs only when either:
    - settings.app.environment == 'development', OR
    - environment variable FORCE_PRELOAD is set to true (1/true/yes)

    We avoid preloading on constrained cloud hosts (e.g., Render) by default.
    """
    try:
        settings = get_settings()
        env = settings.app.environment
        force = os.getenv("FORCE_PRELOAD", "false").lower() in ("1", "true", "yes")

        should_preload = force or (env == "development")

        if not should_preload:
            print(f"Skipping model preload (env={env}, FORCE_PRELOAD={force})")
            return

        print("Preloading embedding model and LLM (this may take a while)...")
        # Import here to avoid importing heavy ML libs at module import time
        from src.ingestion.embeddings import embedding_manager
        from src.llm.llm_factory import get_llm

        # Preload embeddings
        try:
            embedding_manager.get_embeddings()
            print("Embedding model preloaded")
        except Exception as e:
            print(f"Warning: failed to preload embeddings: {e}")

        # Preload LLM
        try:
            get_llm()
            print("✓ LLM preloaded")
        except Exception as e:
            print(f"Warning: failed to preload LLM: {e}")

        print("Model preload complete")
    except Exception as e:
        print(f"Startup preload encountered an unexpected error: {e}")

import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port 
    )

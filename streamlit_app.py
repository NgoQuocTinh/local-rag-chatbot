"""
Streamlit UI for local RAG Chatbot

Features:
- Upload PDFs into data directory
- Run ingestion (overwrite/append) into Chroma DB
- Chat with citations and optional retrieved-context inspection
"""

from __future__ import annotations

import sys
import time
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama

from config.setting import get_settings
from src.ingestion.pdf_processor import PDFProcessor
from src.ingestion.embeddings import embedding_manager
from src.retrieval.retriever import AdvancedRetriever
from src.chat.conversation import ConversationMemory
from src.chat.prompts import get_conversation_prompt
from src.utils.metrics import metrics_tracker


def _project_root() -> Path:
    return Path(__file__).parent


# Ensure imports work when launched via `streamlit run`
sys.path.insert(0, str(_project_root()))


def _ensure_dirs():
    settings = get_settings()
    Path(settings.paths.data_dir).mkdir(parents=True, exist_ok=True)


def _list_pdfs(data_dir: Path) -> List[Path]:
    settings = get_settings()
    seen: set[Path] = set()
    files: List[Path] = []
    for ext in settings.pdf.allowed_extensions:
        for p in data_dir.glob(f"*{ext}"):
            # Windows: .pdf và .PDF trùng file → tránh đếm 2 lần
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                files.append(p)
    return sorted(files)


def _save_uploaded_pdfs(files, data_dir: Path) -> List[Path]:
    saved: List[Path] = []
    for f in files:
        target = data_dir / f.name
        target.write_bytes(f.getbuffer())
        saved.append(target)
    return saved


@st.cache_resource(show_spinner=False)
def _get_embeddings():
    return embedding_manager.get_embeddings()


@st.cache_resource(show_spinner=False)
def _get_vectorstore() -> Chroma:
    settings = get_settings()
    embeddings = _get_embeddings()
    return Chroma(
        persist_directory=settings.paths.db_dir,
        embedding_function=embeddings,
        collection_name=settings.vectordb.collection_name,
    )


@st.cache_resource(show_spinner=False)
def _get_llm() -> ChatOllama:
    settings = get_settings()
    cfg = settings.llm
    return ChatOllama(
        model=cfg.model,
        base_url=cfg.ollama.base_url,
        temperature=cfg.temperature,
        num_predict=cfg.max_tokens,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        repeat_penalty=cfg.repeat_penalty,
        timeout=cfg.ollama.timeout,
    )


def _format_docs(docs) -> str:
    if not docs:
        return "Không tìm thấy thông tin liên quan."
    parts = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "N/A")
        parts.append(f"[Đoạn {i} - File: {filename}, Trang: {page}]\n{doc.page_content}\n")
    return "\n".join(parts)


def _get_memory() -> ConversationMemory:
    if "memory" not in st.session_state or not isinstance(st.session_state.memory, ConversationMemory):
        st.session_state.memory = ConversationMemory()
    return st.session_state.memory


def _run_ingestion(mode: str) -> Tuple[bool, str]:
    """
    mode: 'overwrite' | 'append'
    """
    settings = get_settings()
    db_path = Path(settings.paths.db_dir)

    processor = PDFProcessor()
    chunks = processor.process_directory()
    if not chunks:
        return False, "Không có documents/chunks để xử lý. Kiểm tra thư mục data_dir và file PDF."

    embeddings = _get_embeddings()

    if db_path.exists():
        if mode == "overwrite":
            shutil.rmtree(db_path)
        elif mode != "append":
            return False, f"Mode không hợp lệ: {mode}"

    if db_path.exists() and mode == "append":
        vectordb = Chroma(
            persist_directory=str(db_path),
            embedding_function=embeddings,
            collection_name=settings.vectordb.collection_name,
        )
        vectordb.add_documents(chunks)
        count = vectordb._collection.count()
        return True, f"Đã append dữ liệu. Tổng vectors: {count}"

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(db_path),
        collection_name=settings.vectordb.collection_name,
        collection_metadata={
            "hnsw:space": settings.vectordb.hnsw.space,
            "hnsw:construction_ef": settings.vectordb.hnsw.construction_ef,
            "hnsw:M": settings.vectordb.hnsw.M,
        },
    )
    count = vectordb._collection.count()
    return True, f"Đã tạo database mới. Tổng vectors: {count}"


def main():
    st.set_page_config(page_title="Local RAG Chatbot", layout="wide")
    _ensure_dirs()

    settings = get_settings()

    st.title("Local RAG Chatbot")
    st.caption("Upload PDF → Ingest vào ChromaDB → Chat kèm trích dẫn nguồn.")

    with st.sidebar:
        st.subheader("Cấu hình đang dùng")
        st.write(f"- **Data dir**: `{settings.paths.data_dir}`")
        st.write(f"- **DB dir**: `{settings.paths.db_dir}`")
        st.write(f"- **Embeddings**: `{settings.embeddings.model_name}` ({settings.embeddings.device})")
        st.write(f"- **Chunking**: size={settings.chunking.chunk_size}, overlap={settings.chunking.chunk_overlap}")
        st.write(
            f"- **Retrieval**: type={settings.retrieval.search_type}, k={settings.retrieval.k}, "
            f"threshold={settings.retrieval.score_threshold}"
        )
        if settings.retrieval.search_type == "mmr":
            st.write(
                f"- **MMR**: fetch_k={settings.retrieval.mmr.fetch_k}, lambda={settings.retrieval.mmr.lambda_mult}"
            )
        st.write(f"- **LLM**: `{settings.llm.model}` @ `{settings.llm.ollama.base_url}`")

        st.divider()
        if st.button("Xóa lịch sử chat", use_container_width=True):
            mem = _get_memory()
            mem.clear()
            st.session_state.chat_messages = []
            st.rerun()

    tab_docs, tab_ingest, tab_chat = st.tabs(["Tài liệu (PDF)", "Ingest", "Chat"])

    with tab_docs:
        st.subheader("Quản lý PDF")
        data_dir = Path(settings.paths.data_dir)

        uploaded = st.file_uploader(
            "Upload PDF vào data_dir",
            type=["pdf", "PDF"],
            accept_multiple_files=True,
        )
        if uploaded:
            saved = _save_uploaded_pdfs(uploaded, data_dir)
            st.success(f"Đã lưu {len(saved)} file vào `{data_dir}`")

        files = _list_pdfs(data_dir)
        if not files:
            st.info(f"Chưa có PDF trong `{data_dir}`")
        else:
            st.write(f"Tìm thấy **{len(files)}** file:")
            for p in files:
                st.write(f"- `{p.name}` ({p.stat().st_size / (1024 * 1024):.2f} MB)")

    with tab_ingest:
        st.subheader("Ingest vào ChromaDB")

        db_dir = Path(settings.paths.db_dir)
        exists = db_dir.exists()
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(f"- **DB path**: `{db_dir}`")
            st.write(f"- **DB exists**: `{exists}`")
        with col2:
            mode = st.selectbox("Chế độ", ["overwrite", "append"], index=0 if exists else 0)

        if st.button("Chạy ingest", type="primary"):
            with st.spinner("Đang ingest..."):
                start = time.time()
                ok, msg = _run_ingestion(mode)
                elapsed = time.time() - start

            if ok:
                st.success(f"{msg} (took {elapsed:.2f}s)")
                # Vectorstore cache depends on persisted DB; clear to reflect new DB
                _get_vectorstore.clear()
            else:
                st.error(msg)

    with tab_chat:
        st.subheader("Chat")

        db_dir = Path(settings.paths.db_dir)
        if not db_dir.exists():
            st.warning("Chưa có database. Hãy chạy Ingest trước.")
            return

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        show_context = st.checkbox("Hiển thị ngữ cảnh đã retrieve (để kiểm chứng)", value=False)

        for m in st.session_state.chat_messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        question = st.chat_input("Nhập câu hỏi của bạn...")
        if not question:
            return

        mem = _get_memory()
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Đang trả lời..."):
                start = time.time()

                vectorstore = _get_vectorstore()
                retriever = AdvancedRetriever(vectorstore)
                llm = _get_llm()
                prompt = get_conversation_prompt()

                docs = retriever.retrieve(question)
                context = _format_docs(docs)
                history = mem.get_history_string() if settings.chat.history_enabled else "Không có lịch sử."

                messages = prompt.format_messages(context=context, history=history, question=question)
                llm_result = llm.invoke(messages)
                answer = getattr(llm_result, "content", str(llm_result))

                if settings.chat.show_sources:
                    answer += retriever.format_sources(docs, settings.chat.source_format)

                elapsed = time.time() - start

                mem.add_user_message(question)
                mem.add_assistant_message(answer)

                if settings.metrics.enabled:
                    # Similarity score here is approximate; we track response-time reliably.
                    metrics_tracker.track_query(
                        query=question,
                        response_time=elapsed,
                        retrieval_time=0.0,
                        generation_time=0.0,
                        num_retrieved_docs=len(docs),
                        avg_similarity_score=0.0,
                        success=True,
                    )

                st.markdown(answer)

                if show_context:
                    with st.expander("Ngữ cảnh retrieve"):
                        st.text(context)

        st.session_state.chat_messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()


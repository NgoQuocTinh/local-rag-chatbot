## local-rag-chatbot

Hệ thống **RAG Chatbot chạy local**: ingest PDF → lưu vào **ChromaDB** → chat hỏi đáp dựa trên tài liệu, có **trích dẫn nguồn** và **metrics**. Hỗ trợ đổi LLM linh hoạt (Ollama local, Gemini, và có thể mở rộng thêm provider khác).

---

## Tính năng chính

- **Ingestion (PDF → chunks → embeddings → ChromaDB)**: nhiều file PDF, chunking có overlap, metadata theo file/trang.
- **Retrieval**: `similarity` hoặc **MMR** (đa dạng hoá context), có `score_threshold`.
- **Chat**: hội thoại có memory (sliding window), prompt tiếng Việt, trích dẫn nguồn theo file/trang.
- **LLM linh hoạt**: chọn `llm.provider: ollama | gemini` qua config (không sửa code).
- **UI local**: upload PDF, ingest, chat + xem “retrieved context” để kiểm chứng.
- **Metrics**: lưu vào `metrics.json` (thời gian retrieve/generate/total, số docs, avg similarity khi dùng similarity).

---

## Kiến trúc (tóm tắt luồng)

### Ingest

1. Đọc PDF theo trang (LangChain `PyPDFLoader`)
2. Chunking (`RecursiveCharacterTextSplitter`)
3. Embedding (SentenceTransformers qua `langchain-huggingface`)
4. Lưu vectors vào ChromaDB (HNSW)

### Chat (RAG)

1. Query → retrieve top-k docs (MMR hoặc similarity theo config)
2. Format docs thành `{context}` + thêm `{history}`
3. Prompt → LLM → Answer
4. Append “Nguồn tham khảo” (file/trang)
5. Ghi metrics

---

## Cấu trúc thư mục

- **`config/`**
  - `config.yaml`: cấu hình hệ thống
  - `setting.py`: Pydantic models + loader (hỗ trợ `config/config.{env}.yaml`)
- **`src/ingestion/`**
  - `pdf_processor.py`: tìm/đọc PDF, chunking, metadata
  - `embeddings.py`: load embeddings (lazy + cache)
- **`src/retrieval/retriever.py`**
  - `AdvancedRetriever`: retrieval (MMR/similarity) + citations
  - `retrieve_unified`: cơ chế retrieve thống nhất cho chain + metrics
- **`src/chat/`**
  - `prompts.py`: prompt templates
  - `conversation.py`: sliding-window memory
- **`src/llm/llm_factory.py`**
  - `get_llm()`: tạo LLM theo `llm.provider` (ollama/gemini)
- **Entry points**
  - `ingest.py`: tạo/append vector DB
  - `chat.py`: chat CLI
  - `streamlit_app.py`: UI local

---

## Cài đặt

### 1) Tạo môi trường và cài dependencies

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Cấu hình environment variables

Tạo file `.env` (tham khảo `.env.example`).

- **Dùng Gemini**: cần `GOOGLE_API_KEY`
- **Dùng Ollama**: cần Ollama server chạy local

---

## Cấu hình (`config/config.yaml`)

### Các phần quan trọng

- **`paths.data_dir`**: thư mục chứa PDF (`data/documents`)
- **`paths.db_dir`**: thư mục ChromaDB (`chroma_db`)
- **`chunking`**: `chunk_size`, `chunk_overlap`, `separators`
- **`embeddings`**: `model_name`, `device`, `batch_size`, `normalize`
- **`retrieval`**:
  - `search_type: mmr | similarity`
  - `k`, `score_threshold`
  - `mmr.fetch_k`, `mmr.lambda_mult`
- **`llm`**: chọn provider/model và cấu hình provider
- **`chat`**: history, `source_format`, messages
- **`metrics`**: bật/tắt, `save_interval`

### Dùng nhiều “profile” theo use case

`config/setting.py` hỗ trợ override theo env:

- `RAG_ENVIRONMENT=development` → tự động merge `config/config.development.yaml` (nếu có)
- Bạn có thể tạo thêm: `config/config.study.yaml`, `config/config.invoices.yaml`, …

---

## Chạy hệ thống (CLI)

### 1) Ingest PDF → ChromaDB

1. Copy PDF vào `paths.data_dir` (mặc định `data/documents`)
2. Chạy:

```bash
python ingest.py
```

Nếu DB đã tồn tại, CLI sẽ hỏi `overwrite` hoặc `append`.

### 2) Chat trong terminal

```bash
python chat.py
```

Lệnh hỗ trợ:
- `clear`: xoá lịch sử hội thoại
- `stats`: xem thống kê metrics
- `help`: hướng dẫn
- `exit`: thoát

---

## Chạy hệ thống (UI)

```bash
streamlit run streamlit_app.py
```

Trong UI:
- Upload PDF vào `paths.data_dir`
- Chạy ingest (overwrite/append)
- Chat + xem citations + (tuỳ chọn) xem “retrieved context” để kiểm chứng

---

## Chuyển đổi LLM (Ollama ↔ Gemini)

### Ollama (local)

1. Bật Ollama server:

```bash
ollama serve
```

2. Trong `config.yaml`:

```yaml
llm:
  provider: "ollama"
  model: "llama3"
  ollama:
    base_url: "http://localhost:11434"
```

### Gemini

1. Set `.env`:

```bash
GOOGLE_API_KEY=your-key
```

2. Trong `config.yaml`:

```yaml
llm:
  provider: "gemini"
  model: "gemini-2.5-flash"
```

---

## Kiểm chứng “có bịa không?”

- Bật `chat.show_sources: true` và `chat.source_format: detailed`
- Dùng UI bật “Hiển thị ngữ cảnh đã retrieve” để xem đúng đoạn gốc
- Khi nghi ngờ, hỏi tiếp: “Trích nguyên văn đoạn trong tài liệu dùng để trả lời, kèm file + trang.”

---

## Troubleshooting nhanh

- **Không thấy PDF**: kiểm tra `paths.data_dir` và đặt file `.pdf` đúng thư mục.
- **Gemini không chạy**: thiếu `GOOGLE_API_KEY` trong `.env`.
- **Ollama không chạy**: chưa `ollama serve` hoặc sai `base_url`.
- **CUDA embeddings lỗi**: đổi `embeddings.device: "cpu"` để test.
- **DB rỗng**: chạy lại `ingest.py` và kiểm tra log số chunks/vectors.

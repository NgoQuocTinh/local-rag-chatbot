# local-rag-chatbot

## Run (CLI)

- **Ingest PDFs into ChromaDB**:

```bash
python ingest.py
```

- **Chat (terminal)**:

```bash
python chat.py
```

## Run (UI)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start UI:

```bash
streamlit run streamlit_app.py
```

3. In the UI:
- Upload PDF files into your configured `paths.data_dir`
- Run **Ingest** (overwrite/append)
- Chat and inspect retrieved context + citations
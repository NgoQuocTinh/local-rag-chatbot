# Personal Knowledge Assistant (Local AI Chatbot)

Welcome to your **Personal Knowledge Assistant**! This is a secure, private, and smart note-taking application powered by AI. You write notes, and the AI reads them so you can ask it questions later. 

It is designed to run entirely on your own computer (a desktop-first approach), meaning your data is 100% safe and never sent to the cloud unless you specifically choose to.

---

## Key Features

- **100% Private (Local-First):** Your Markdown notes and database are stored directly on your computer's hard drive. No hidden cloud syncs.
- **Smart Note-Taking:** A clean, easy-to-use editor with auto-formatting for your personal notes.
- **Knowledge Graph:** Visually explore how different notes and ideas connect to each other.
- **"Chat with your Notes" (RAG):** Ask the AI anything about your documents, and it will answer while providing exact citations (saying exactly which file it got the information from).
- **Flexible AI Integration:** You can run completely free and offline AI models (like Ollama) or plug in advanced online models (like Google Gemini).
- **Zero-Delay Startup:** The AI preloads into your computer's memory when you open the app, ensuring your very first chat message is answered instantly without long loading bars.

---

## Project Structure (Keep it simple)

This system is divided into two main pieces that talk to each other:

- **The Brain (Backend):** Built with Python. It reads your notes, organizes them into a searchable database (ChromaDB), and talks to the AI.
- **The Interface (Frontend):** Built with React/Next.js. This is the visual application you interact with (the sidebar, the editor, the chat window, and the graph).

---

## How to Run the Application

To start the app, you need to turn on both the "Brain" and the "Interface" using two separate terminals (command prompts).

### Step 1: Start the Brain (Backend)

1. Open your terminal and go to the project folder.
2. Activate your Python environment and install the required tools:
```powershell
# Create and activate a virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r backend/requirements.txt
```
3. Copy the `backend/.env.example` file and rename it to `backend/.env`. (If you use Gemini, put your API key here).
4. Run the server and tell the AI to load immediately (`FORCE_PRELOAD="true"`):
```powershell
$env:FORCE_PRELOAD="true"
python backend/server.py
```
*(The backend brain is now running in the background at `http://localhost:8000`)*

### Step 2: Start the Interface (Frontend)

1. Open a **NEW** terminal and go into the frontend folder:
```powershell
cd frontend
```
2. Install the necessary web packages:
```powershell
npm install
```
3. Start the application:
```powershell
npm run dev
```

🎉 **You're done!** Open your web browser and go to `http://localhost:3000`. You can start writing notes and chatting with your AI Assistant immediately.

---

## Customizing Your AI

You can easily change how the AI behaves by editing the `backend/config/config.yaml` file:

- **Switching Models:**
  ```yaml
  llm:
    provider: "ollama"  # Change this to "gemini" if you want to use Google's model
    model: "llama3"     # Change this to "gemini-1.5-flash" if using Gemini
  ```

## The Future (Desktop App)
Right now, this runs as a web page on your local computer. The ultimate goal of this project is to bundle it into a **standalone Desktop Application** (an `.exe` on Windows or `.dmg` on Mac) using tools like **Tauri** or **Electron**. Once bundled, you'll just double-click an icon to launch it like Notion or Obsidian, without ever needing to touch a terminal!
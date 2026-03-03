# RARE — Research Agent with Retrieval and Explanation

<p align="center">
  <strong>An AI-powered research assistant with a multi-agent council, RAG pipeline, and web search — wrapped in a sleek dark-themed UI.</strong>
</p>

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multi-Agent Council** | Multiple AI agents (powered by Groq) discuss your question in parallel, then a Judge synthesizes the best answer |
| **RAG Pipeline** | Upload documents → chunked with LangChain → embedded with SentenceTransformers → indexed with FAISS → context-aware answers |
| **Web Search** | Real-time web search integration for up-to-date information |
| **Gemini Summarizer** | Google Gemini-powered document summarization |
| **User Auth** | Secure registration & login with Argon2 password hashing + JWT tokens |
| **Chat History** | Persistent chat sessions stored in MongoDB, with create/delete support |
| **Dark UI** | Premium "RARE" black theme with cyan accents, smooth animations, and responsive design |

---

## 🏗️ Architecture

```
RiyaAI-main/
├── backend/                  # FastAPI server
│   ├── app.py                # Main API — auth, chat, RAG, council endpoints
│   ├── agents/
│   │   └── agent.py          # Multi-agent council logic
│   ├── services/
│   │   ├── groq_api.py       # Groq LLM client with token tracking
│   │   ├── rag.py            # RAG retrieval logic
│   │   ├── document_store.py # FAISS vector store management
│   │   ├── search_service.py # Web search integration
│   │   ├── gemini_summarizer.py  # Google Gemini summarizer
│   │   ├── huggingface_api.py    # HuggingFace inference
│   │   └── openrouter_api.py     # OpenRouter fallback
│   ├── models/
│   │   └── schemas.py        # Pydantic models
│   ├── DockerFile            # Container config for deployment
│   ├── requirements.txt      # Python dependencies
│   └── .env                  # API keys (not committed)
│
├── frontend/                 # Vite + React SPA
│   ├── src/
│   │   ├── App.jsx           # Root component with auth & routing
│   │   ├── LoginPage.jsx     # Login/Register page
│   │   ├── Sidebar.jsx       # Chat history sidebar
│   │   ├── MainContent.jsx   # Chat interface & message display
│   │   ├── CouncilView.jsx   # Council deliberation viewer
│   │   ├── App.css           # Global styles & design tokens
│   │   ├── LoginPage.css     # Login page styles
│   │   ├── Sidebar.css       # Sidebar styles
│   │   ├── MainContent.css   # Chat interface styles
│   │   └── CouncilView.css   # Council view styles
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── .gitignore
└── README.md                 # ← You are here
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** and **npm**
- **MongoDB** — local instance or [MongoDB Atlas](https://www.mongodb.com/atlas) connection string

### 1. Clone the Repository

```bash
git clone https://github.com/akshitsrivastava04/RARE-Research-Agent-with-Retrieval-and-Explaination.git
cd RARE-Research-Agent-with-Retrieval-and-Explaination
```

### 2. Backend Setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Configure Environment Variables

Create a `.env` file inside `backend/`:

```env
GROQ_API_KEY="your_groq_api_key"
GOOGLE_API_KEY="your_google_gemini_api_key"
HUGGINGFACEHUB_API_TOKEN="your_hf_token"
OPENROUTER_API_KEY="your_openrouter_key"
MONGO_DETAILS="mongodb://localhost:27017"
SECRET_KEY="generate_a_random_secret_key"
```

> **Tip:** Generate a secret key with `python -c "import secrets; print(secrets.token_hex(32))"`

#### Start the Backend

```bash
uvicorn app:app --reload
```

The API will be available at **http://localhost:8000**. Docs at `/docs`.

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server
npm run dev
```

The frontend will be available at **http://localhost:5173**.

---

## 🐳 Docker (Backend Only)

```bash
# From the project root
docker build -f backend/DockerFile -t rare-backend .
docker run -p 8080:8080 --env-file backend/.env rare-backend
```

---

## 🔑 API Keys You'll Need

| Service | Get it at | Used for |
|---|---|---|
| **Groq** | [console.groq.com](https://console.groq.com) | Primary LLM inference (Llama, Qwen, etc.) |
| **Google Gemini** | [aistudio.google.com](https://aistudio.google.com) | Document summarization |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Embedding models (optional) |
| **OpenRouter** | [openrouter.ai](https://openrouter.ai) | Fallback LLM provider (optional) |

---

## 🧪 Testing

```bash
cd backend
python test_council.py
```

This runs a test of the multi-agent council system to verify your API keys and agent configuration are working.

---

## 📝 Tech Stack

**Backend:** FastAPI · Uvicorn · Motor (async MongoDB) · FAISS · SentenceTransformers · LangChain · Groq SDK · Google Generative AI  
**Frontend:** React 19 · Vite · Framer Motion · React Markdown · Axios  
**Auth:** Argon2 hashing · JWT (python-jose)  
**Deployment:** Docker · Gunicorn

---

## 📄 License

This project is open-source. Feel free to fork and build upon it.

---

<p align="center">
  Built with ☕ by <a href="https://github.com/akshitsrivastava04">Akshit Srivastava</a>
</p>

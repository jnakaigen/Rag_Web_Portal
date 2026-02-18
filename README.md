# 🧠 RAG Intelligence Portal

A Retrieval-Augmented Generation (RAG) web application that allows users to chat with their own data. The system ingests content from **Websites (URLs)** or **PDF Documents**, processes them into vector embeddings, and uses a Generative AI model to answer queries based strictly on the uploaded context.

## 🚀 Features

* **Dual Data Sources:** Scrape text from websites or parse uploaded PDF documents.
* **Smart Chunking:** Splits large documents into manageable chunks with metadata.
* **Vector Search:** Uses `Sentence-Transformers` and Cosine Similarity to find relevant context.
* **Precision Filtering:** Applies a similarity threshold (0.25) to ignore irrelevant data.
* **Generative AI:** Integrates with LLMs (via OpenRouter/Gemini) to generate natural answers.
* **Modern UI:** A clean React frontend with live status updates and source citations.

## 🛠️ Tech Stack

* **Frontend:** React (Vite), CSS3
* **Backend:** FastAPI (Python), Uvicorn
* **AI & ML:** PyTorch, Sentence-Transformers, OpenAI Client (for OpenRouter)
* **Data Processing:** BeautifulSoup4 (Web), PyPDF2 (PDF)

---

## 📋 Prerequisites

Before running the project, ensure you have the following installed:
* **Python 3.10+**
* **Node.js & npm**
* An API Key from [OpenRouter](https://openrouter.ai/) (or OpenAI).

---

## ⚙️ Installation Guide

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-project-folder>

### 2. Backend Setup (Python)
Open a terminal in the root directory:

Bash
# Create a virtual environment (Optional but recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
Create a .env file in the root directory and add your API key:

Code snippet
OPENROUTER_API_KEY=sk-or-v1-your-actual-api-key-here
### 3. Frontend Setup (React)
Open a new terminal window and navigate to the frontend folder:

Bash
cd frontend
npm install
🏃‍♂️ How to Run
You need to run the Backend and Frontend in two separate terminals.

Terminal 1: Start Backend API
Bash
# Make sure you are in the root folder
uvicorn main:app --reload
The API will start at http://127.0.0.1:8000

Terminal 2: Start Frontend UI
Bash
# Make sure you are in the frontend folder
cd frontend
npm run dev
The UI will start at http://localhost:5173

📖 Usage
Open the web portal (usually http://localhost:5173).

Add Data:

Paste a URL (e.g., a Wikipedia page) and click "Scrape Site".

Or, upload a PDF document and click "Process PDF".

Wait for Processing: The sidebar will show "Processing..." and then "Success!" once the embeddings are stored.

Ask Questions: Type a query in the chat box.

Example: "What is the summary of the document I uploaded?"

View Sources: Click the "📚 View Sources" dropdown under the AI's answer to see exactly which text chunks were used.

📂 Project Structure
Plaintext
├── main.py              # FastAPI Backend Entry Point
├── data_ingestion.py    # Logic for scraping URLs and parsing PDFs
├── embedding.py         # VectorStore class and embedding generation
├── rag_engine.py        # Core RAG logic (Retrieval + Generation)
├── requirements.txt     # Python dependencies
├── vector_store.pt      # (Auto-generated) Stores the vector database
├── .env                 # API Keys (Not committed to Git)
└── frontend/            # React Frontend Folder
    ├── src/
    │   ├── App.jsx      # Main UI Logic
    │   └── App.css      # Styling
    ├── package.json     # Frontend dependencies
    └── vite.config.js   # Vite configuration
🛡️ License
This project is developed for academic submission.


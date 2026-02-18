from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import torch
import shutil

# Import your existing logic
from data_ingestion import scrape_url, parse_pdf, chunk_text
from embedding import generate_embeddings, VectorStore
from rag_engine import retrieve_top_k, generate_answer
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow React to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # React's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State (The Brain)
store = VectorStore()
DB_FILE = "vector_store.pt"

# Load existing brain on startup
if os.path.exists(DB_FILE):
    try:
        state = torch.load(DB_FILE)
        store.add_data(state["chunks"], state["embeddings"])
        print(f"Loaded {len(store.chunks)} chunks.")
    except Exception as e:
        print(f"Failed to load DB: {e}")

# --- Data Models ---
class UrlRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

# --- API Endpoints ---

@app.get("/")
def home():
    return {"status": "active", "docs": len(store.chunks)}

@app.post("/ingest/url")
def ingest_url(request: UrlRequest):
    try:
        raw_text = scrape_url(request.url)
        if "Error" in raw_text:
            raise HTTPException(status_code=400, detail=raw_text)
            
        chunks = chunk_text(raw_text, chunk_size=500, source=request.url)
        vectors = generate_embeddings(chunks)
        store.add_data(chunks, vectors)
        
        # Save to disk
        torch.save({"chunks": store.chunks, "embeddings": store.embeddings}, DB_FILE)
        
        return {"message": f"Successfully added {len(chunks)} chunks from URL."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        # Save temp file
        with open(f"temp_{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Parse
        with open(f"temp_{file.filename}", "rb") as f:
            raw_text = parse_pdf(f.read())
            
        # Cleanup
        os.remove(f"temp_{file.filename}")
        
        chunks = chunk_text(raw_text, chunk_size=500, source=file.filename)
        vectors = generate_embeddings(chunks)
        store.add_data(chunks, vectors)
        
        torch.save({"chunks": store.chunks, "embeddings": store.embeddings}, DB_FILE)
        
        return {"message": f"Successfully added {len(chunks)} chunks from PDF."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat(request: QueryRequest):
    if not store.chunks:
        raise HTTPException(status_code=400, detail="Database is empty. Add documents first.")
    
    # 1. Retrieve
    relevant_chunks = retrieve_top_k(request.query, store, k=5)
    
    # 2. Generate
    if not os.getenv("OPENROUTER_API_KEY"):
        answer = "No API Key found. Showing context only."
    else:
        answer = generate_answer(request.query, relevant_chunks)
        
    return {
        "answer": answer,
        "sources": relevant_chunks
    }

@app.post("/reset")
def reset_db():
    global store
    store = VectorStore()
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    return {"message": "Knowledge base cleared."}
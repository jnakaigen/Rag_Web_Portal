import streamlit as st
import os
import torch
import time

# Import your logic
from data_ingestion import scrape_url, parse_pdf, chunk_text
from embedding import generate_embeddings, VectorStore
from rag_engine import retrieve_top_k, generate_answer

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="RAG Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
    .stChatInput {position: fixed; bottom: 0; padding-bottom: 20px;}
    .block-container {padding-top: 2rem; padding-bottom: 10rem;}
    h1 {color: #2e86c1;}
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE SETUP ---
if "store" not in st.session_state:
    st.session_state.store = VectorStore()
    if os.path.exists("vector_store.pt"):
        try:
            state = torch.load("vector_store.pt")
            st.session_state.store.add_data(state["chunks"], state["embeddings"])
        except:
            pass

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. SIDEBAR (Ingestion Only) ---
with st.sidebar:
    st.title("🧠 Knowledge Base")
    st.markdown("---")
    
    # API key box removed from here! 
    # It now uses the key from your .env / rag_engine.py automatically.

    st.subheader("Add Data")
    tab1, tab2 = st.tabs(["🌐 Web", "📄 PDF"])
    
    # URL Tab
    with tab1:
        url_input = st.text_input("Website URL", placeholder="https://example.com")
        if st.button("Scrape Site", type="primary"):
            if not url_input:
                st.warning("Please enter a URL")
            else:
                with st.status("Reading website...", expanded=True) as status:
                    st.write("Fetching content...")
                    raw_text = scrape_url(url_input)
                    if "Error" in raw_text:
                        status.update(label="Failed", state="error")
                        st.error(raw_text)
                    else:
                        st.write("Chunking text...")
                        chunks = chunk_text(raw_text, chunk_size=500, source=url_input)
                        st.write("Generating AI embeddings...")
                        vectors = generate_embeddings(chunks)
                        st.session_state.store.add_data(chunks, vectors)
                        
                        torch.save({
                            "chunks": st.session_state.store.chunks,
                            "embeddings": st.session_state.store.embeddings
                        }, "vector_store.pt")
                        
                        status.update(label="Knowledge Added!", state="complete", expanded=False)
                        st.toast(f"Added {len(chunks)} chunks from URL", icon="✅")

    # PDF Tab
    with tab2:
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        if uploaded_file and st.button("Process PDF", type="primary"):
            with st.status("Processing PDF...", expanded=True) as status:
                st.write("Extracting text...")
                raw_text = parse_pdf(uploaded_file.read())
                
                st.write("Chunking & Embedding...")
                chunks = chunk_text(raw_text, chunk_size=500, source=uploaded_file.name)
                vectors = generate_embeddings(chunks)
                st.session_state.store.add_data(chunks, vectors)
                
                torch.save({
                    "chunks": st.session_state.store.chunks,
                    "embeddings": st.session_state.store.embeddings
                }, "vector_store.pt")
                
                status.update(label="PDF Ingested!", state="complete", expanded=False)
                st.toast(f"Added {len(chunks)} chunks from PDF", icon="📄")

    st.markdown("---")
    doc_count = len(st.session_state.store.chunks)
    st.metric("Total Memories", doc_count)
    
    if st.button("🗑️ Reset Brain"):
        if os.path.exists("vector_store.pt"):
            os.remove("vector_store.pt")
        st.session_state.store = VectorStore()
        st.session_state.messages = []
        st.rerun()

# --- 5. MAIN CHAT INTERFACE ---
st.title("🤖 Intelligent Agent")
st.caption("Ask questions based on your uploaded documents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for src in message["sources"]:
                    st.markdown(f"**From:** *{src['metadata'].get('source', 'Unknown')}*")
                    st.info(src['text'])

if query := st.chat_input("Ask me anything about your data..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if not st.session_state.store.chunks:
        response = "🚫 **Brain Empty!** Please add some data in the sidebar first."
        with st.chat_message("assistant"):
            st.error(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("🧠 *Thinking...*")
            
            # Step A: Retrieve
            relevant_chunks = retrieve_top_k(query, st.session_state.store, k=5)
            
            # Step B: Generate (Automatically uses rag_engine's API logic)
            ai_response = generate_answer(query, relevant_chunks)
            
            thinking_placeholder.markdown(ai_response)
            
            with st.expander("📚 View Context Sources"):
                for c in relevant_chunks:
                    st.caption(f"Source: {c['metadata'].get('source', 'Unknown')} | Score: {c['score']:.2f}")
                    st.text(c['text'])

        st.session_state.messages.append({
            "role": "assistant", 
            "content": ai_response,
            "sources": relevant_chunks 
        })
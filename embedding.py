from sentence_transformers import SentenceTransformer
import torch
import os
# 1. INITIALIZE MODEL
# We load the model once to avoid reloading it every time we process text.
# 'all-MiniLM-L6-v2' is a lightweight model perfect for CPU use.
print("Loading Embedding Model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_embeddings(chunks):
    """
    Takes a list of text chunks and converts them into vector embeddings.
    
    Args:
        chunks (list): A list of dictionaries, where each dict has a "text" key.
        
    Returns:
        torch.Tensor: A matrix of vectors (one vector per chunk).
    """
    # Extract just the text content from our dictionary objects
    text_list = [chunk['text'] for chunk in chunks]
    
    # Generate embeddings
    # convert_to_tensor=True makes it easier to use with PyTorch later
    embeddings = model.encode(text_list, convert_to_tensor=True)
    
    return embeddings

class VectorStore:
    """
    A simple in-memory database to store our RAG data.
    In a real app, this would be replaced by ChromaDB or Pinecone.
    """
    def __init__(self):
        self.chunks = []       # Stores the actual text and metadata
        self.embeddings = None # Stores the mathematical vectors
        
    def add_data(self, new_chunks, new_embeddings):
        """
        Adds new data to the store, combining it with existing data.
        """
        self.chunks.extend(new_chunks)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            # Concatenate the new vectors to the existing list of vectors
            self.embeddings = torch.cat((self.embeddings, new_embeddings), dim=0)

# --- TERMINAL TEST BLOCK ---
# This runs only if you execute: python embedding.py
if __name__ == "__main__":
    import json
    import numpy as np
    from data_ingestion import scrape_url, chunk_text

    SEPARATOR = "-" * 60
    INPUT_FILE = "my_chunks.json"   # The text file you created in Step 1
    OUTPUT_FILE = "vector_store.pt" # The new "Brain" file we will create

    print(f"\n{SEPARATOR}")
    print(" 1. LOADING TEXT DATA ")
    print(SEPARATOR)

    chunks = []
    
    # 1. Load the text chunks
    if os.path.exists(INPUT_FILE):
        print(f"Reading from '{INPUT_FILE}'...")
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"✅ Loaded {len(chunks)} chunks.")
    else:
        print(f"❌ '{INPUT_FILE}' not found. Please run data_ingestion.py first.")
        exit()

    print(f"\n{SEPARATOR}")
    print(" 2. GENERATING EMBEDDINGS ")
    print(SEPARATOR)
    
# 2. Convert to Vectors
    print(f"Converting text to vectors (this might take a moment)...")
    vectors = generate_embeddings(chunks)
    
    print("✅ Conversion Complete.")
    print(f"Shape: {vectors.shape}")

    print(f"\n{SEPARATOR}")
    print(" 3. SAVING TO FILE ")
    print(SEPARATOR)
    
    # 3. Save EVERYTHING (Text + Math) to one file
    print(f"Saving database to '{OUTPUT_FILE}'...")
    
    # We create a dictionary containing both pieces of the puzzle
    db_state = {
        "chunks": chunks,      # The readable text
        "embeddings": vectors  # The search vectors
    }
    
    torch.save(db_state, OUTPUT_FILE)
    
    print(f"✅ Success! Data saved to '{OUTPUT_FILE}'.")
    print(f"{SEPARATOR}\n")
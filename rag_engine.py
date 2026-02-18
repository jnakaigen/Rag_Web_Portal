import os
import torch
from sentence_transformers import util
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
# We import the model and VectorStore class from your previous file
# This ensures we use the exact same logic for both saving and loading
from embedding import model, VectorStore

# --- 1. SETUP LLM CLIENT ---
# We use OpenRouter to access DeepSeek (free) as per your notebook reference.
# You can replace this with any OpenAI-compatible provider.
# If you don't have a key, the code will still run the search part successfully.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY") or "YOUR_API_KEY_HERE"
)

def retrieve_top_k(query, vector_store, k=5, threshold=0.25):
    """
    1. Embeds the user query.
    2. Compares it with all stored vectors using Cosine Similarity.
    3. Returns the top k most relevant text chunks (IF they pass the threshold).
    """
    # Safety check: Is the store empty?
    if vector_store.embeddings is None:
        return []
        
    # 1. Convert query to vector
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # 2. Calculate Similarity (Query vs. All Chunks)
    scores = util.cos_sim(query_embedding, vector_store.embeddings)[0]
    
    # 3. Pick Top K
    k = min(k, len(vector_store.chunks))
    top_results = torch.topk(scores, k=k)
    
    # 4. Package results with THRESHOLD FILTER
    relevant_chunks = []
    for score, idx in zip(top_results.values, top_results.indices):
        # --- NEW: Similarity Threshold Check ---
        if score.item() < threshold:
            continue
            
        chunk = vector_store.chunks[idx]
        relevant_chunks.append({
            "score": score.item(),
            "text": chunk['text'],
            "metadata": chunk['metadata']
        })
        
    return relevant_chunks

def generate_answer(query, context_chunks):
    """
    Constructs the carefully constructed prompt and sends it to the LLM.
    """
    if not context_chunks:
        return "I couldn't find any relevant information in your documents to answer that question."

    # Combine all retrieved text into one big string with citations
    context_text = "\n\n".join(
        [f"[Source: {c['metadata'].get('source', 'Doc')}]\n{c['text']}" for c in context_chunks]
    )
    
    # Create the Carefully Constructed Prompt
    prompt = f"""
    You are an expert academic assistant. Your goal is to answer the user query based ONLY on the context provided below.
    
    GUIDELINES:
    1. If the answer is not in the context, state "I do not know based on the provided documents."
    2. Keep the answer structured and professional.

    CONTEXT:
    {context_text}
    
    USER QUERY: 
    {query}
    """
    
    try:
        # Call the AI Model
        response = client.chat.completions.create(
            model="openrouter/free", # Change this if using a different model
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {e}\n(Did you set your API Key?)"

# --- TERMINAL TEST BLOCK ---
if __name__ == "__main__":
    SEPARATOR = "-" * 60
    INPUT_FILE = "vector_store.pt" # The file you created in embedding.py

    print(f"\n{SEPARATOR}")
    print(" 1. LOADING BRAIN (Vector Store) ")
    print(SEPARATOR)
    
    store = VectorStore()
    
    if os.path.exists(INPUT_FILE):
        print(f"Loading '{INPUT_FILE}'...")
        # Load the dictionary we saved earlier
        state = torch.load(INPUT_FILE)
        # Restore the data into our class
        store.add_data(state["chunks"], state["embeddings"])
        print(f"✅ Success! Brain loaded with {len(store.chunks)} memories.")
    else:
        print(f"❌ '{INPUT_FILE}' not found.")
        print("Please run embedding.py first to generate the data.")
        exit()

    print(f"\n{SEPARATOR}")
    print(" 2. TESTING RETRIEVAL (Semantic Search) ")
    print(SEPARATOR)
    
    # Since we scraped "Artificial Intelligence" in the previous step, let's ask about it.
    query = "What are the main threats to the Amazon rainforest?" 
    print(f"Query: \"{query}\"")
    
    # Get top 3 results
    results = retrieve_top_k(query, store, k=3)
    
    print(f"\nTop {len(results)} Retrieved Results:")
    for i, r in enumerate(results):
        print(f"  Result {i+1} (Score: {r['score']:.4f}): \"{r['text'][:100]}...\"")
        
    print(f"\n{SEPARATOR}")
    print(" 3. TESTING GENERATION (LLM Answer) ")
    print(SEPARATOR)
    
    # Check if we can actually call the API
    if "YOUR_API_KEY_HERE" in client.api_key and not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  No API Key detected. Skipping actual LLM call.")
        print("To see the AI answer, set your API key in the script or environment.")
        print(f"\n[Prompt Preview]:\n...Context: {results[0]['text'][:50]}...\nUser Query: {query}")
    else:
        print("Sending prompt to AI...")
        answer = generate_answer(query, results)
        print(f"\n🤖 AI Response:\n{answer}")

    print(f"{SEPARATOR}\n")
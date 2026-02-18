import requests
from bs4 import BeautifulSoup
import PyPDF2
import io
import json  # <--- Add this at the top

def save_chunks_to_file(chunks, filename="processed_data.json"):
    """Saves the list of chunks to a JSON file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)
        return f"Saved {len(chunks)} chunks to '{filename}'"
    except Exception as e:
        return f"Error saving file: {e}"
def scrape_url(url):
    """Fetches and parses text from a URL."""
    try:
        # Send a fake user-agent so websites don't block us
        headers = {"User-Agent": "Mozilla/5.0"} 
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Error check
        
        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract all paragraph text
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        return f"Error fetching URL: {e}"

def parse_pdf(file_bytes):
    """Extracts text from a PDF file object."""
    try:
        # Create a PDF reader object from the bytes
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        # Loop through every page and extract text
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error parsing PDF: {e}"

def chunk_text(text, chunk_size=500, source="Unknown"):
    """Splits text into chunks and adds metadata."""
    chunks = []
    # Simple loop to slice text by character count
    for i in range(0, len(text), chunk_size):
        chunk_content = text[i:i+chunk_size]
        # Store as a dictionary to keep text and metadata together
        chunks.append({
            "text": chunk_content,
            "metadata": {"source": source, "start_index": i}
        })
    return chunks

# Example usage:
if __name__ == "__main__":
    import os
    
    # Define a separator for clean output
    SEPARATOR = "-" * 60

    # --- TEST 1: URL SCRAPING ---
    print(f"\n{SEPARATOR}")
    print(" 1. TESTING URL SCRAPING ")
    print(SEPARATOR)
    
    test_url = "https://en.wikipedia.org/wiki/Amazon_rainforest"
    print(f"Target URL: {test_url}")
    
    url_text = scrape_url(test_url)
    if url_text and "Error" not in url_text:
        print("Status:     Success ✅")
        print(f"Text Length: {len(url_text)} characters")
        print(f"\n[Raw Text Preview]:\n\"{url_text[:200]}...\"")
    else:
        print(f"Status:     Failed ❌\nOutput: {url_text}")

    # --- TEST 2: PDF EXTRACTION ---
    print(f"\n{SEPARATOR}")
    print(" 2. TESTING PDF EXTRACTION ")
    print(SEPARATOR)

    test_pdf_name = "Amazon_rainforest.pdf"
    
    if os.path.exists(test_pdf_name):
        print(f"Target PDF: {test_pdf_name}")
        try:
            with open(test_pdf_name, "rb") as f:
                pdf_bytes = f.read()
                pdf_text = parse_pdf(pdf_bytes)
            
            print("Status:     Success ✅")
            print(f"Text Length: {len(pdf_text)} characters")
            print(f"\n[Raw PDF Text Preview]:\n\"{pdf_text[:200]}...\"")
        except Exception as e:
            print(f"Status:     Failed ❌ ({e})")
    else:
        print(f"Target PDF: {test_pdf_name}")
        print("Status:     Skipped ⚠️ (File not found)")
        print("Tip: Place a file named 'sample.pdf' in this folder to test.")
        # Use URL text as fallback for chunking test
        pdf_text = url_text 

    # --- TEST 3: CHUNKING LOGIC ---
    print(f"\n{SEPARATOR}")
    print(" 3. TESTING CHUNKING LOGIC ")
    print(SEPARATOR)
    
    # We use the text from the successful method (URL or PDF)
    source_text = pdf_text if 'pdf_text' in locals() and len(pdf_text) > 0 else url_text
    
    chunk_size = 200
    chunks = chunk_text(source_text, chunk_size=chunk_size, source="Test_Source")
    
    print(f"Chunk Size:   {chunk_size} characters")
    print(f"Total Chunks: {len(chunks)}")
    
    print("\n[First 5 Chunks Preview]:")
    for i in range(min(5, len(chunks))):
        # Remove newlines for cleaner single-line display
        chunk_content = chunks[i]["text"].replace("\n", " ") 
        print(f"  Chunk {i+1:02}: \"{chunk_content[:80]}...\"")

    print(f"{SEPARATOR}\n")
    # ... previous code ...

    print(f"{SEPARATOR}\n")
    
    # --- TEST 4: SAVING DATA ---
    print(f"{SEPARATOR}")
    print(" 4. SAVING PROCESSED DATA ")
    print(SEPARATOR)
    
    save_status = save_chunks_to_file(chunks, "my_chunks.json")
    print(save_status)
    print(f"{SEPARATOR}\n")
    

# scrape_url: Uses requests to download the webpage and BeautifulSoup to strip away HTML tags, leaving only the human-readable text inside <p> (paragraph) tags.

# parse_pdf: Uses PyPDF2 to read a raw binary file (like an uploaded PDF), iterates through its pages, and concatenates the text.

# chunk_text: This is crucial for RAG. LLMs can't read entire books at once. We split the text into smaller pieces (e.g., 500 characters). We store each piece as a dictionary containing the text and metadata (like where it came from), satisfying Sub-Task 1.
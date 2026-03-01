import os
import pickle
import re
from rank_bm25 import BM25Okapi

# --- CONFIG INTEGRATION ---
from backend.core.config import DATA_DIR, BM25_INDEX_PATH

# --- MODULE IMPORTS ---
from backend.core.chunking import chunk_pdf
from backend.database.vector_store import VectorManager

def multilingual_tokenize(text):
    """
    Research-Grade Lightweight Multilingual Tokenizer.
    - English: Extracts alphanumeric words (e.g., "hello", "world", "2024").
    - Japanese: Extracts individual characters (Kanji, Hiragana, Katakana).
    This solves the BM25 "whitespace" problem for CJK languages without 
    requiring heavy NLP libraries like MeCab.
    """
    text = text.lower()
    
    # REGEX BREAKDOWN:
    # [a-z0-9]+                 -> Matches 1 or more English letters/numbers
    # \u4e00-\u9fff             -> Matches Kanji (Chinese characters used in JP)
    # \u3040-\u309f             -> Matches Hiragana
    # \u30a0-\u30ff             -> Matches Katakana
    pattern = r'[a-z0-9]+|[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]'
    
    return re.findall(pattern, text)

def save_bm25_index(chunks):
    """
    Creates and saves a BM25 (Keyword) Index for Hybrid Search.
    """
    print("   [BM25] Tokenizing corpus (Multilingual Mode)...")
    
    # We pass the content of every chunk through our custom regex tokenizer
    tokenized_corpus = [multilingual_tokenize(chunk["content"]) for chunk in chunks]
    
    print("   [BM25] Building Index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # We save the BM25 object AND the chunks (to map ID -> Text later)
    # NOTE: For future scale, saving the full chunk here is redundant 
    # since Milvus also stores it, but it is excellent for rapid prototyping.
    data_to_save = {
        "bm25_obj": bm25,
        "chunk_map": {chunk["chunkID"]: chunk for chunk in chunks}
    }
    
    print(f"   [BM25] Saving index to {BM25_INDEX_PATH}...")
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(data_to_save, f)
    print("   [BM25] Saved successfully.")

def main():
    # 1. Define input file
    pdf_path = os.path.join(DATA_DIR, "content.pdf")
    
    # 2. PHASE 1: CHUNKING
    print("\n=== PHASE 1: CHUNKING & EXTRACTION ===")
    chunks_with_metadata = chunk_pdf(pdf_path)
    
    if not chunks_with_metadata:
        print("Pipeline aborted: No chunks generated.")
        return

    # 3. PHASE 2: KEYWORD INDEXING (BM25)
    print("\n=== PHASE 2: BUILDING KEYWORD INDEX (BM25) ===")
    save_bm25_index(chunks_with_metadata)

    # 4. PHASE 3: VECTORIZATION & STORAGE
    print("\n=== PHASE 3: VECTORIZATION & STORAGE ===")
    vector_db = VectorManager()
    vector_db.embed_and_store(chunks_with_metadata)
    
    print("\n=== PROCESS COMPLETE ===")
    print(f"System ready. Processed {len(chunks_with_metadata)} chunks.")

if __name__ == "__main__":
    main()
import os
import sys
import pickle
import numpy as np
import re
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from langdetect import detect
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIG INTEGRATION ---
from backend.core.config import (
    DB_PATH, 
    COLLECTION_NAME, 
    EMBEDDING_MODEL_NAME,
    DEVICE,
    BM25_INDEX_PATH,
    WEIGHT_VECTOR,
    WEIGHT_KEYWORD, # Set to 0.2 (20%) in config
    INITIAL_RETRIEVAL_K,
    FINAL_TOP_K
)

# --- SETUP GEMINI FOR TRANSLATION ---
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
translation_model = genai.GenerativeModel('gemini-2.5-flash-lite')

# --- GLOBAL CACHE ---
# Prevents reloading the .pkl file from the hard drive on every search
_BM25_CACHE = None

def get_bm25_data():
    global _BM25_CACHE
    if _BM25_CACHE is None:
        try:
            with open(BM25_INDEX_PATH, "rb") as f:
                _BM25_CACHE = pickle.load(f)
        except FileNotFoundError:
            print("Error: BM25 Index not found. Run embedding.py first.")
            return None, None
    return _BM25_CACHE["bm25_obj"], _BM25_CACHE["chunk_map"]

def multilingual_tokenize(text):
    """Tokenizes English by word, and Japanese/Chinese by character."""
    text = text.lower()
    pattern = r'[a-z0-9]+|[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]'
    return re.findall(pattern, text)

def translate_query(query, target_lang):
    """Translates the user query into the document's native language."""
    print(f"🔄 Translating query to '{target_lang.upper()}'...")
    prompt = f"Translate the following search query to {target_lang.upper()}. Output ONLY the translated text, nothing else. Query: '{query}'"
    try:
        response = translation_model.generate_content(prompt)
        translated_text = response.text.strip()
        print(f"✅ Translated Query: '{translated_text}'")
        return translated_text
    except Exception as e:
        print(f"⚠️ Translation failed, using original query. Error: {e}")
        return query

def normalize_scores(score_dict):
    """Min-Max Normalization: (x - min) / (max - min)"""
    if not score_dict:
        return {}
    scores = list(score_dict.values())
    min_score, max_score = min(scores), max(scores)
    if max_score == min_score:
        return {k: 1.0 for k in score_dict}
    return {k: (v - min_score) / (max_score - min_score) for k, v in score_dict.items()}

def hybrid_search(user_query_text):
    print(f"\n--- 🗣️ User Input: '{user_query_text}' ---")
    
    # ==========================================
    # 0. PRE-FLIGHT: LANGUAGE DETECTION & TRANSLATION
    # ==========================================
    bm25, chunk_map = get_bm25_data()
    if not bm25 or not chunk_map:
        return []

    # Grab doc_lang from the first available chunk
    first_chunk_key = list(chunk_map.keys())[0]
    doc_lang = chunk_map[first_chunk_key].get("doc_lang", "en").lower()
    
    # Detect User Query Language
    try:
        user_lang = detect(user_query_text).lower()
    except:
        user_lang = "en"
        
    print(f"📄 Document Language: {doc_lang.upper()}")
    print(f"👤 User Query Language: {user_lang.upper()}")

    # The Translation Bridge
    search_query = user_query_text
    if user_lang != doc_lang:
        search_query = translate_query(user_query_text, doc_lang)
    
    # ==========================================
    # 1. VECTOR SEARCH (Semantic)
    # ==========================================
    print(f"\n1. Executing Vector Search (Weight: {WEIGHT_VECTOR})...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    query_vector = model.encode([search_query], normalize_embeddings=True)[0].tolist()
    
    client = MilvusClient(uri=DB_PATH)
    vector_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=INITIAL_RETRIEVAL_K,
        output_fields=["chunk_id"]
    )
    
    vec_scores = {}
    if vector_results:
        for res in vector_results[0]:
            vec_scores[res['entity']['chunk_id']] = res['distance']
            
    # ==========================================
    # 2. KEYWORD SEARCH (Exact Match - 20%)
    # ==========================================
    print(f"2. Executing BM25 Search (Weight: {WEIGHT_KEYWORD})...")
    query_tokens = multilingual_tokenize(search_query)
    raw_bm25_scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(raw_bm25_scores)[::-1][:INITIAL_RETRIEVAL_K]
    
    bm25_scores_dict = {}
    map_keys = list(chunk_map.keys())
    for idx in top_indices:
        if idx < len(map_keys):
            chunk_id = chunk_map[map_keys[idx]]['chunkID']
            score = raw_bm25_scores[idx]
            if score > 0: 
                bm25_scores_dict[chunk_id] = score

    # ==========================================
    # 3. NORMALIZATION & FUSION
    # ==========================================
    print("3. Normalizing and Fusing Scores...")
    norm_vec = normalize_scores(vec_scores)
    norm_bm25 = normalize_scores(bm25_scores_dict)
    
    all_ids = set(norm_vec.keys()) | set(norm_bm25.keys())
    final_scores = []
    
    for cid in all_ids:
        v_score = norm_vec.get(cid, 0.0)
        k_score = norm_bm25.get(cid, 0.0)
        final_score = (v_score * WEIGHT_VECTOR) + (k_score * WEIGHT_KEYWORD)
        
        final_scores.append({
            "chunk_id": cid,
            "score": final_score,
            "vector_score": v_score,
            "keyword_score": k_score
        })
    
    final_scores.sort(key=lambda x: x["score"], reverse=True)
    top_results = final_scores[:FINAL_TOP_K]

    # Quick terminal print (Optional, can be removed to clean up Streamlit logs)
    print(f"\n🏆 Top {len(top_results)} Hybrid Matches Found.")
    return top_results

if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is this document about?"
    hybrid_search(query)
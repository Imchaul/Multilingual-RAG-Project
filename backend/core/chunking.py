import os
import sys
from transformers import AutoTokenizer
from langdetect import detect

# Imports
from backend.core.extraction import extract_all_spans, analyze_chunk_metadata, count_tokens
from backend.core.config import TOKENIZER_NAME, MAX_TOKENS, CHUNK_OVERLAP, active_regex_decider

# --- 1. SETUP TOKENIZER ---
print(f"Loading Tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def process_chunk(text, cursor, all_spans, chunk_count, doc_lang):
    """Helper to format a single chunk"""
    if not text.strip():
        return None, cursor
        
    token_size = count_tokens(text, tokenizer)
    metadata, new_cursor = analyze_chunk_metadata(text, all_spans, cursor)
    
    return {
        "isHeading": metadata["isHeading"],
        "heading": metadata["heading"],
        "chunkID": chunk_count + 1,
        "pageNo": metadata["pageNo"],
        "token_size": token_size,
        "content": text,
        "doc_lang": doc_lang  # <--- Saved for Cross-Lingual RAG
    }, new_cursor

def chunk_pdf(pdf_path):
    print(f"--- Chunking PDF: {pdf_path} ---")
    
    # 1. Extraction (Join with Newlines to preserve structure)
    try:
        all_spans = extract_all_spans(pdf_path)
        full_content = "\n".join([span["text"] for span in all_spans])
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

    # 1.2 DETECT DOCUMENT LANGUAGE
    # We sample text from the middle of the document to avoid cover pages/indexes
    sample_text = full_content[1000:6000] if len(full_content) > 6000 else full_content
    try:
        doc_lang = detect(sample_text)
    except:
        doc_lang = "EN" # Fallback if detection fails (e.g., all numbers)
        
    print(f"🌍 Detected Document Language: '{doc_lang.upper()}'")
    ACTIVE_REGEX_PATTERN = active_regex_decider(doc_lang)

    # 2. Splitting
    sentences = ACTIVE_REGEX_PATTERN.split(full_content)

    # 3. Chunking Loop
    chunks_with_metadata = []
    current_token_count = 0
    current_chunk_text = ""
    span_search_cursor = 0 
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence: continue

        sentence_token_count = count_tokens(sentence, tokenizer)

        # ======================================================
        # CASE A: THE "BOULDER" (> 512 Tokens)
        # We ONLY use overlap here, inside this specific block.
        # ======================================================
        if sentence_token_count > MAX_TOKENS:
            # Clean newlines for a neat log
            clean_sent = sentence.replace('\n', ' ')
            
            # Grab first 100 and last 50 characters
            first_part = clean_sent[:100]
            last_part = clean_sent[-50:]
            snippet = f"{first_part} ... {last_part}"
            
            print(f"⚠️  Handling massive block ({sentence_token_count} tokens) via Overlap...")
            print(f"   🔍 Snippet: '{snippet}'")
            
            # 1. Flush whatever small text we had pending (Zero Overlap)
            if current_chunk_text:
                chunk_data, span_search_cursor = process_chunk(
                    current_chunk_text, span_search_cursor, all_spans, len(chunks_with_metadata), doc_lang
                )
                if chunk_data: chunks_with_metadata.append(chunk_data)
                current_chunk_text = ""
                current_token_count = 0

            # 2. Slice the "Boulder" using a Rolling Window (Overlap APPLIED HERE)
            token_ids = tokenizer.encode(sentence, add_special_tokens=False)
            total_ids = len(token_ids)
            start_idx = 0
            
            while start_idx < total_ids:
                end_idx = min(start_idx + MAX_TOKENS, total_ids)
                window_tokens = token_ids[start_idx:end_idx]
                window_text = tokenizer.decode(window_tokens, skip_special_tokens=True)
                
                chunk_data, span_search_cursor = process_chunk(
                    window_text, span_search_cursor, all_spans, len(chunks_with_metadata), doc_lang
                )
                if chunk_data: chunks_with_metadata.append(chunk_data)
                
                if end_idx == total_ids: break
                    
                # The "Stride" determines the overlap
                # (Next chunk starts 50 tokens *back* from the end of this one)
                stride = MAX_TOKENS - CHUNK_OVERLAP
                start_idx += stride
            
            continue 

        # ======================================================
        # CASE B: NORMAL SENTENCES (< 512 Tokens)
        # Just stack them. No overlap logic here.
        # ======================================================
        if current_token_count + sentence_token_count <= MAX_TOKENS:
            current_chunk_text += sentence + " "
            current_token_count += sentence_token_count
        else:
            # Bucket Full -> Save current text
            chunk_data, span_search_cursor = process_chunk(
                current_chunk_text, span_search_cursor, all_spans, len(chunks_with_metadata), doc_lang
            )
            if chunk_data:
                chunks_with_metadata.append(chunk_data)

            # Start new bucket with current sentence (ZERO OVERLAP)
            current_chunk_text = sentence + " "
            current_token_count = sentence_token_count

    # Final flush
    if current_chunk_text:
        chunk_data, span_search_cursor = process_chunk(
            current_chunk_text, span_search_cursor, all_spans, len(chunks_with_metadata), doc_lang
        )
        if chunk_data:
            chunks_with_metadata.append(chunk_data)
            
    print(f"Generated {len(chunks_with_metadata)} chunks.")
    return chunks_with_metadata

if __name__ == "__main__":
    from backend.core.config import DATA_DIR
    test_pdf = os.path.join(DATA_DIR, "content.pdf")
    results = chunk_pdf(test_pdf)

    # ==========================================
    # RESULT PREVIEWER
    # ==========================================
    print("\n" + "="*60)
    print(f"🎉 CHUNKING COMPLETE! Total Chunks Generated: {len(results)}")
    print("Previewing the first 5 chunks...")
    print("="*60)

    # Change [:5] to a larger number if you want to see more chunks
    for chunk in results[:5]:
        print(f"\n📦 Chunk ID: {chunk['chunkID']} | Page: {chunk['pageNo']} | Tokens: {chunk['token_size']} | Lang: {chunk['doc_lang'].upper()}")
        
        if chunk['isHeading']:
            print(f"🏷️  Heading : {chunk['heading']}")
        else:
            print(f"🏷️  Heading : None (Standard Text)")
            
        # Clean newlines for a neat terminal output
        clean_content = chunk['content'].replace('\n', ' ')
        
        # If the text is short, just print it all. 
        # Otherwise, print first 100 and last 100.
        if len(clean_content) <= 200:
            print(f"📄 Content : {clean_content}")
        else:
            first_100 = clean_content[:100]
            last_100 = clean_content[-100:]
            print(f"📄 Content : {first_100}\n   ... [SNIPPED] ... \n   {last_100}")
            
    print("\n" + "="*60)
import fitz  # PyMuPDF
from collections import Counter
from transformers import AutoTokenizer

# --- CONFIG INTEGRATION ---
from backend.core.config import BOLD_FONT_INDICATORS

def count_tokens(text, tokenizer):
    """Returns the number of tokens in a text string."""
    # add_special_tokens=False ensures we just count words, not [CLS]/[SEP] yet
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)


def is_font_bold(font_name, flags):
    """
    Detects boldness by checking BOTH the PDF flag and the Font Name.
    """
    if flags & 16:
        return True
    
    name_lower = font_name.lower()
    BOLD_FONT_INDICATORS = ["bold", "bd", "demi", "black", "heavy", "w6", "w7", "w8", "w9", "h"]
    
    for indicator in BOLD_FONT_INDICATORS:
        if indicator in name_lower:
            return True
    return False

def is_font_italic(font_name, flags):
    """
    Detects italics by checking the PDF flag (bit 1) and the Font Name.
    Works perfectly even if the font is also bolded.
    """
    # Bit 1 (value 2) represents Italic in PyMuPDF
    if flags & 2:
        return True
    
    name_lower = font_name.lower()
    # Common strings found in italic/oblique font names
    ITALIC_FONT_INDICATORS = ["italic", "it", "oblique", "obl"]
    
    for indicator in ITALIC_FONT_INDICATORS:
        if indicator in name_lower:
            return True
            
    return False

def extract_all_spans(pdf_path):
    """
    Scans the PDF and returns a flat list of ALL text spans with metadata.
    """
    doc = fitz.open(pdf_path)
    all_spans = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        
                        is_bold = is_font_bold(span["font"], span["flags"])
                        is_italic = is_font_italic(span["font"], span["flags"])
                        
                        all_spans.append({
                            "text": text,
                            "size": round(span["size"], 2),
                            "font": span["font"],
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "page": page_num + 1
                        })
    doc.close()
    return all_spans

def get_style_score(span):
    """
    Assigns a priority score based on styling:
    3 = Bold + Italic
    2 = Bold
    1 = Italic
    0 = Regular
    """
    if span.get("is_bold") and span.get("is_italic"):
        return 3
    elif span.get("is_bold"):
        return 2
    elif span.get("is_italic"):
        return 1
    return 0

def analyze_chunk_metadata(chunk_text, all_spans, cursor):
    """
    Analyzes a chunk of text against the global spans to find the best heading.
    Priority: Size > Rarity > Style (Bold+Italic > Bold > Italic > Regular).
    """
    chunk_spans = []
    chunk_text_clean = chunk_text.replace(" ", "").replace("\n", "")
    matched_text = ""
    new_cursor = cursor
    
    # 1. Gather all spans that make up this specific chunk
    while new_cursor < len(all_spans) and len(matched_text) < len(chunk_text_clean):
        span = all_spans[new_cursor]
        span_clean = span["text"].replace(" ", "").replace("\n", "")
        matched_text += span_clean
        chunk_spans.append(span)
        new_cursor += 1

    if not chunk_spans:
        return {"isHeading": False, "heading": "", "pageNo": 1}, cursor

    # Default fallback page number
    primary_page = chunk_spans[0]["page"]

    # 2. Group spans by Font Size
    size_groups = {}
    for span in chunk_spans:
        size = span["size"]
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(span)

    # 3. Sort sizes descending (Size Matters First)
    sorted_sizes = sorted(size_groups.keys(), reverse=True)
    
    total_spans_in_chunk = len(chunk_spans)
    winning_size = None

    # 4. Find the largest size that is "Rare"
    # (e.g., it makes up less than 30% of the chunk)
    for size in sorted_sizes:
        spans_of_this_size = size_groups[size]
        frequency = len(spans_of_this_size) / total_spans_in_chunk
        
        if frequency < 0.30:  # <--- Adjust this rarity threshold if needed
            winning_size = size
            break

    # If no rare text is found, there is no heading in this chunk
    if winning_size is None:
        return {"isHeading": False, "heading": "", "pageNo": primary_page}, new_cursor

    # 5. The Tie-Breaker (Style Priority)
    # Get all spans that have the winning font size
    candidate_spans = size_groups[winning_size]
    
    # Use our get_style_score function to find the absolute best span
    best_span = max(candidate_spans, key=get_style_score)

    # Extract the winning text
    heading_text = best_span["text"].strip()
    
    # If the heading is too short (like a stray number or bullet), ignore it
    if len(heading_text) < 3:
        return {"isHeading": False, "heading": "", "pageNo": primary_page}, new_cursor

    return {
        "isHeading": True, 
        "heading": heading_text, 
        "pageNo": best_span["page"]
    }, new_cursor
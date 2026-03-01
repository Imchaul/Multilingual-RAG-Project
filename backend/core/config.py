import os
import re

# ==========================================
# 1. PATH SETTINGS
# ==========================================
# Automatically find the project root (ITSOVER/) no matter where this runs
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define key directories
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(BASE_DIR, "milvus_demo.db") # Local Milvus file

# ==========================================
# 2. MODEL SETTINGS
# ==========================================
# We use BGE-M3 because it is SOTA for Multilingual RAG
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
TOKENIZER_NAME = "intfloat/multilingual-e5-small"

# ==========================================
# 2. DEVICE SETTINGS
# ==========================================
DEVICE = "cuda"

# Vector Dimension (Must match the model!)
# BGE-M3 = 1024, OpenAI = 1536
VECTOR_DIMENSION = 384

# ==========================================
# 3. CHUNKING PARAMETERS
# ==========================================
MAX_TOKENS = 512
CHUNK_OVERLAP = 50
UTILIZATION_THRESHOLD = 0.8  # If chunk < 80% full, apply overlap logic

# ==========================================
# 4. SPLITTING LOGIC (SCALABILITY CORE)
# ==========================================
# Pattern A: English Only (Strict, looks for Capital Letters after dots)
# Use this for pure English papers like "Attention Is All You Need"
REGEX_SPLIT_EN = r'(?<=[.!?])\s+(?=[A-Z])'

# Pattern B: Smart Multilingual (JP + EN)
# Use this for Japanese Docs or Mixed content
# Splits on: Japanese periods (。), Exclamations, or English dots NOT preceded by a digit
REGEX_SPLIT_JA = r'([。！？]|(?<!\d)\.(?=\s)|\.$)'

# --- ACTIVE CONFIGURATION ---
# Change this variable to switch your entire pipeline's behavior!
def active_regex_decider(doc_lang):
    if doc_lang == "en":
        return re.compile(REGEX_SPLIT_EN)
    elif doc_lang == "ja":
        return re.compile(REGEX_SPLIT_JA)

# ==========================================
# 5. METADATA & EXTRACTION SETTINGS
# ==========================================
# Keywords to detect if a font is "Bold" (covers English & Japanese conventions)
BOLD_FONT_INDICATORS = [
    "bold", "bd", "demi", "black", "heavy",  # English Standard
    "w6", "w7", "w8", "w9",                  # Japanese Standard (Weights)
    "h"                                      # "Heavy"
]

# Minimum text length to consider a PDF "Digital" (vs Scanned)
MIN_TEXT_FOR_DIGITAL_CHECK = 50

# ==========================================
# 6. DATABASE SETTINGS
# ==========================================
COLLECTION_NAME = "rag_collection"
METRIC_TYPE = "COSINE"

# ==========================================
# 7. SEARCH & RANKING SETTINGS
# ==========================================
# Hybrid Search Weights (Must sum to 1.0)
WEIGHT_VECTOR = 0.8  # 80% Semantic
WEIGHT_KEYWORD = 0.2  # 20% Exact Match

# Path to save the BM25 Index (The "Keyword Brain")
BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")

# How many results to retrieve from EACH method before fusing?
# (We grab more than needed, then filter down to top 5)
INITIAL_RETRIEVAL_K = 50 
FINAL_TOP_K = 5
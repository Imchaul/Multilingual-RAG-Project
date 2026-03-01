import os
import sys
import pickle
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# ==========================================
# 1. PATH SETUP (CRITICAL)
# ==========================================
# Look 1 level up to find the 'backend' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# 2. BACKEND IMPORTS 
# ==========================================
from backend.core.config import BM25_INDEX_PATH
from backend.core.search import hybrid_search 

# ==========================================
# 3. CONFIGURATION & SETUP
# ==========================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="RAG Scholar", page_icon="🎓", layout="wide")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env file! Please add it.")
    st.stop()

# Configure Gemini 1.5 Flash (Optimized for fast, factual RAG)
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# ==========================================
# 4. HELPER: LOAD CHUNK MAP MANUALLY
# ==========================================
@st.cache_resource
def load_chunk_map():
    """Loads the pickle file ONCE to get the mapping of ID -> Text."""
    if not os.path.exists(BM25_INDEX_PATH):
        return None
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
        return data["chunk_map"]

# ==========================================
# 5. UI & CHAT LOGIC
# ==========================================
st.title("🎓 Research Assistant (Hybrid RAG)")
st.caption("Powered by Milvus + BM25 + Gemini 1.5 Flash")

# Load context map
chunk_map = load_chunk_map()
if not chunk_map:
    st.error(f"⚠️ Index file not found at {BM25_INDEX_PATH}. Please run 'python -m backend.core.embedding' first.")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if prompt := st.chat_input("Ask a question about the document..."):
    # 1. Show User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Retrieve Context (Hybrid Search)
    context_text = ""
    with st.spinner("🔍 Searching knowledge base (Auto-Translating if needed)..."):
        try:
            # Call your backend search
            results = hybrid_search(prompt)
            
            # Extract content from chunk_map using O(1) direct lookup
            context_blocks = []
            if results:
                for res in results:
                    chunk_id = res['chunk_id']
                    
                    # O(1) Lookup: Instantly fetch the dictionary instead of looping
                    data = chunk_map.get(chunk_id)
                    if data:
                        header = f"Page {data['pageNo']} ({data['heading']})"
                        block = f"**[{header}]**\n{data['content']}"
                        context_blocks.append(block)
                
                context_text = "\n\n".join(context_blocks)
            
        except Exception as e:
            st.error(f"Search System Error: {e}")

    # 3. Generate Answer
    if context_text:
        system_prompt = f"""
        You are an expert bilingual research assistant (English/Japanese). 
        Your task is to answer the user's question based strictly on the provided search results.

        🔍 **Context Data:**
        {context_text}

        📝 **Guidelines:**
        1. **Language:** Answer in the SAME language as the user's question.
        2. **Accuracy:** Use ONLY the provided context. Do not invent facts.
        3. **Citations:** Mention the [Page Number] or [Heading] if available in the context.
        4. **Uncertainty:** If the exact answer is missing, state what IS known, then politely say you cannot find the rest.
           - (EN): "I couldn't find specific details on X, but the document mentions..."
           - (JP): "Xに関する具体的な記述は見当たりませんが、文書には..."

        ❓ **User Question:**
        {prompt}
        """
        
        try:
            with st.spinner("🤖 Generating answer..."):
                generation_config = genai.types.GenerationConfig(
                    temperature=0.1, # Keep it extremely factual
                )
                response = model.generate_content(system_prompt, generation_config=generation_config)
                answer = response.text
        except Exception as e:
            answer = f"⚠️ LLM Error: {e}"
    else:
        answer = "I couldn't find any relevant context in the document database."

    # 4. Show Assistant Response
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("View Retrieved Context (Source Data)"):
            st.markdown(context_text)

    st.session_state.messages.append({"role": "assistant", "content": answer})
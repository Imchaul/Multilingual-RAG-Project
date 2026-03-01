# 🎓 Hybrid RAG Research Assistant

A lightweight, research-grade Retrieval Augmented Generation (RAG) system capable of **Cross-Lingual Retrieval (English/Japanese)**.

Built to run efficiently on consumer hardware (2GB VRAM), this project implements a custom "Translation Bridge" architecture to solve the semantic gap between Japanese queries and English technical documentation.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Stack](https://img.shields.io/badge/Stack-Milvus_Lite_+_Gemini_Flash-orange) ![Status](https://img.shields.io/badge/Status-Research_Prototype-green)

## 🚀 Key Features

* **🧠 Hybrid Search Architecture**: Fuses **Dense Vector Retrieval** (Multilingual-E5) with **Sparse Keyword Search** (BM25) to capture both semantic meaning and exact technical terms.
* **🌐 Cross-Lingual "Interpreter"**: Automatically detects Japanese queries, translates intent to English for high-precision retrieval, and answers in the user's native language.
* **🧩 Smart "Bucket" Chunking**: A custom sliding-window strategy that respects sentence boundaries in both English (period-based) and Japanese (`。` based) to prevent context loss.
* **⚡ Edge-Optimized**: Designed to run entirely on a laptop with <4GB VRAM using quantized models and Milvus Lite.

## 🛠️ Tech Stack

* **LLM**: Google Gemini 1.5 Flash (via API)
* **Embeddings**: `intfloat/multilingual-e5-small` (SOTA for compact multilingual retrieval)
* **Vector DB**: Milvus Lite (Serverless local vector storage)
* **Framework**: Streamlit (Frontend), PyTorch (Backend)

## 📦 Installation & Setup

### Prerequisites
* Python **3.12** is recommended.
* A Google Gemini API Key (Get one [here](https://aistudio.google.com/)).

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/hybrid-rag-research.git](https://github.com/imchaul/Multilingual-RAG-Project.git)
cd hybrid-rag-research
```

### 2. Add Your Document

Place your PDF inside the `data/` directory located at the root of the repository.

Directory structure:

```
project-root/
│
├── data/
│ └── content.pdf ← Place your document here
```

⚠️ The file **must be named** `content.pdf`.

This document will be processed, chunked, embedded, and indexed into Milvus Lite.

### 3. Generate Embeddings & Build Index

Run the embedding pipeline:

```bash
python -m core.backend.embedding
```

This step will:

- Perform sentence-aware, token-bounded chunking (≤512 tokens)
- Apply sliding-window overlap when necessary
- Generate multilingual dense embeddings (`intfloat/multilingual-e5-small`)
- Build a BM25 sparse index
- Store vectors in Milvus Lite for ANN search
- Prepare hybrid retrieval metadata

Wait until the indexing process completes successfully before moving to the next step.

### 4. Launch the Streamlit Application

Start the frontend interface:

```bash
streamlit run frontend/app.py
```

This will launch a local web application where you can:

- Ask questions in **English or Japanese**
- Trigger hybrid dense–sparse retrieval (80% cosine similarity + 20% BM25)
- Generate grounded responses via **Gemini 1.5 Flash**

## Example Workflow

1. Upload `content.pdf` into the `data/` folder
2. Run the embedding pipeline
3. Launch the Streamlit app
4. Ask your questions

The system will:
- Detect query language
- Translate if document/query languages differ
- Retrieve top-5 hybrid results
- Generate a context-grounded answer

---
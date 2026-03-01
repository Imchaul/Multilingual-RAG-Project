import os
import torch
from tqdm import tqdm  # Progress bar library
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer

# --- CONFIG INTEGRATION ---
from backend.core.config import (
    DB_PATH, 
    COLLECTION_NAME, 
    EMBEDDING_MODEL_NAME, 
    VECTOR_DIMENSION,
    METRIC_TYPE,
    DEVICE  # We use this as a default preference
)

class VectorManager:
    def __init__(self):
        print("\n--- 🧠 Initializing Vector DB Manager ---")
        
        # 1. Connect to Milvus Lite
        print(f"🔌 Connecting to DB at: {DB_PATH}")
        self.client = MilvusClient(uri=DB_PATH)
        
        # 2. Smart Device Selection (Hybrid GPU/CPU)
        # We try to use GPU first. If VRAM runs out, we fall back to CPU.
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ No GPU detected. Defaulting to CPU.")

        # 3. Load Embedding Model (With Safety Fallback)
        print(f"📥 Loading Model: {EMBEDDING_MODEL_NAME}...")
        try:
            # Try loading on the detected device (GPU?)
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
            print(f"✅ Model loaded successfully on {self.device.upper()}")
        except Exception as e:
            # If GPU fails (e.g., Out of Memory), fall back to CPU
            print(f"⚠️ Error loading on {self.device}: {e}")
            print("🔄 Falling back to CPU mode...")
            self.device = "cpu"
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        
        # 4. Ensure Collection Exists
        self._create_collection()

    def _create_collection(self):
        """
        Creates the 'Map' (Schema) for the database if it doesn't exist.
        """
        if self.client.has_collection(COLLECTION_NAME):
            print(f"📂 Collection '{COLLECTION_NAME}' already exists.")
            return

        print(f"🆕 Creating new collection: {COLLECTION_NAME}")
        
        # Define Schema
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        
        # Core Fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
        
        # Metadata Fields
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64)
        schema.add_field(field_name="page_number", datatype=DataType.INT64)
        schema.add_field(field_name="is_heading", datatype=DataType.BOOL)
        schema.add_field(field_name="heading_text", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

        # Indexing
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector", 
            index_type="IVF_FLAT", 
            metric_type=METRIC_TYPE,
            params={"nlist": 128}
        )

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params
        )

    def embed_and_store(self, chunks):
        """
        Takes raw chunks, converts to Normalized Vectors, and saves to DB.
        """
        if not chunks:
            print("❌ No chunks to store.")
            return

        print(f"\n🏗️  Processing {len(chunks)} chunks...")
        
        # 1. Extract text
        texts = [c["content"] for c in chunks]
        
        # 2. Generate Vectors (With Progress Bar!)
        # show_progress_bar=True gives you a nice loading bar for the heavy AI work
        print(f"🧠 Generating Embeddings on {self.device.upper()}...")
        try:
            vectors = self.model.encode(
                texts, 
                normalize_embeddings=True, 
                show_progress_bar=True, 
                batch_size=8 # Small batch size to prevent OOM on small GPUs
            )
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("💥 GPU Out of Memory! Switching to CPU for this batch...")
                self.model.to("cpu")
                vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
            else:
                raise e
        
        # 3. Prepare & Insert Data (With Custom Progress Bar)
        data_rows = []
        print("💾 Preparing Database Records...")
        
        # tqdm wraps the loop to show a progress bar
        for i, chunk in enumerate(tqdm(chunks, desc="Formatting Data")):
            data_rows.append({
                "id": chunk["chunkID"],
                "vector": vectors[i].tolist(),
                "chunk_id": chunk["chunkID"],
                "page_number": chunk["pageNo"],
                "is_heading": chunk["isHeading"],
                "heading_text": chunk["heading"],
                "content": chunk["content"]
            })
            
        # 4. Insert into DB
        print("🚀 Inserting into Milvus...")
        res = self.client.insert(collection_name=COLLECTION_NAME, data=data_rows)
        print(f"✅ Success! Inserted {res['insert_count']} records into Milvus.")
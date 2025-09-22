import sqlite3
import os
import faiss
import numpy as np
from fastembed import TextEmbedding

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.db")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss.index")

def get_chunks():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, text FROM chunks")
    rows = cur.fetchall()
    conn.close()
    return rows

def build_faiss_index():
    chunks = get_chunks()
    ids = [row[0] for row in chunks]
    texts = [row[1] for row in chunks]

    print(f"Total chunks: {len(texts)}")

    # Loaded embedding model
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Generated embeddings
    embeddings = np.array(list(model.embed(texts))).astype("float32")

    # Created FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index + ids mapping
    faiss.write_index(index, INDEX_PATH)
    np.save(INDEX_PATH + ".ids.npy", np.array(ids))

    print("✅ FAISS index built and saved.")

if __name__ == "__main__":
    build_faiss_index()

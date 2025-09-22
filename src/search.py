import os
import sqlite3
import faiss
import numpy as np
from fastembed import TextEmbedding

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.db")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss.index")
IDS_PATH = INDEX_PATH + ".ids.npy"

# Load FAISS index + ids
def load_index():
    index = faiss.read_index(INDEX_PATH)
    ids = np.load(IDS_PATH)
    return index, ids

# Get chunk texts by IDs from SQLite
def get_chunk_texts(ids):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    q_marks = ",".join("?" * len(ids))
    cur.execute(f"SELECT id, doc_name, text FROM chunks WHERE id IN ({q_marks})", tuple(ids))
    results = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    conn.close()
    return results

# Search top-k chunks
def search(query, k=5):
    # Load FAISS + ids
    index, ids = load_index()
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Embed query
    query_vec = np.array(list(model.embed([query]))).astype("float32")

    # Search in FAISS
    distances, indices = index.search(query_vec, k)
    results = []

    # Get corresponding chunk texts
    candidate_ids = [int(i) for i in ids[indices[0]]]
    chunk_map = get_chunk_texts(candidate_ids)

    for rank, (chunk_id, dist) in enumerate(zip(candidate_ids, distances[0]), 1):
        if chunk_id in chunk_map:
            doc, text = chunk_map[chunk_id]
            results.append({
                "rank": rank,
                "doc": doc,
                "score": float(dist),
                "text": text[:300] + "..."  # preview first 300 chars
            })

    return results

if __name__ == "__main__":
    query = "What are common safety measures in industrial machinery?"
    results = search(query, k=3)
    for r in results:
        print(f"[{r['rank']}] {r['doc']} (score={r['score']:.4f})\n{r['text']}\n")

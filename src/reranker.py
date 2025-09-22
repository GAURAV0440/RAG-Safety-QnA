import os
import sqlite3
import faiss
import numpy as np
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi

# Paths
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.db")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss.index")
IDS_PATH = INDEX_PATH + ".ids.npy"

# Loaded FAISS here
def load_index():
    index = faiss.read_index(INDEX_PATH)
    ids = np.load(IDS_PATH)
    return index, ids

# --- Get chunk texts
def get_chunk_texts(ids):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    q_marks = ",".join("?" * len(ids))
    cur.execute(f"SELECT id, doc_name, text FROM chunks WHERE id IN ({q_marks})", tuple(ids))
    results = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    conn.close()
    return results

# ---------- Hybrid Search (FAISS + BM25)
def hybrid_search(query, k=5, alpha=0.6):
    """
    alpha = weight for FAISS semantic score (0-1).
    (1-alpha) = weight for BM25 keyword score.
    """
    index, ids = load_index()
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Embed query
    query_vec = np.array(list(model.embed([query]))).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_vec, k*2)  # take more candidates
    candidate_ids = [int(i) for i in ids[indices[0]]]
    chunk_map = get_chunk_texts(candidate_ids)

    # --- BM25 over candidate texts ---
    texts = [chunk_map[cid][1] for cid in candidate_ids]
    tokenized_corpus = [t.split() for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())

    # --- Normalize scores ---
    faiss_scores = -distances[0]
    faiss_norm = (faiss_scores - faiss_scores.min()) / (faiss_scores.max() - faiss_scores.min() + 1e-9)
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-9)

    # --- Combine ---
    combined_scores = alpha * faiss_norm + (1 - alpha) * bm25_norm

    # --- Rank final ---
    ranked = sorted(zip(candidate_ids, combined_scores), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for rank, (cid, score) in enumerate(ranked, 1):
        doc, text = chunk_map[cid]
        results.append({
            "rank": rank,
            "doc": doc,
            "score": float(score),
            "text": text[:300] + "..."
        })

    return results

if __name__ == "__main__":
    query = "What are common safety measures in industrial machinery?"
    results = hybrid_search(query, k=3, alpha=0.6)
    for r in results:
        print(f"[{r['rank']}] {r['doc']} (score={r['score']:.4f})\n{r['text']}\n")

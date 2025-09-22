import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from search import search
from reranker import hybrid_search
import uvicorn

SOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sources.json")

def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("(", "").replace(")", "")

with open(SOURCE_PATH, "r") as f:
    raw_sources = json.load(f)
SOURCES = {normalize_name(item["title"]): item["url"] for item in raw_sources}

# Threshold for abstaining
SCORE_THRESHOLD = 0.3

app = FastAPI()

class Query(BaseModel):
    q: str
    k: int = 3
    mode: str = "baseline"

@app.post("/ask")
def ask(query: Query):
    # Choose baseline or hybrid search
    if query.mode == "hybrid":
        results = hybrid_search(query.q, k=query.k)
    else:
        results = search(query.q, k=query.k)

    # Check threshold (abstain if no strong enough match)
    if not results or results[0]["score"] < SCORE_THRESHOLD:
        return {
            "answer": None,
            "contexts": results,
            "reranker_used": query.mode,
            "abstained": True
        }
    answer_parts = []
    for idx, r in enumerate(results[:query.k]):
        sentences = r["text"].split(".")
        snippet = ".".join(sentences[:2]).strip()
        if snippet:
            answer_parts.append(f"{snippet} [{idx+1}]")
    answer = " ".join(answer_parts) if answer_parts else None

    # Attach citation URLs using normalized lookup
    for r in results:
        fname = normalize_name(r["doc"])
        r["url"] = SOURCES.get(fname, None)

    return {
        "answer": answer,
        "contexts": results,
        "reranker_used": query.mode,
        "abstained": False
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

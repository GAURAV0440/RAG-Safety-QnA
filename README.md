### RAG-Safety-QnA

This project is a mini RAG (Retrieval-Augmented Generation) + Reranker Q&A service.
It answers questions from a small set of industrial and machinery safety PDFs.

The system:

Splits PDFs into chunks.

Creates vector embeddings and builds a FAISS index.

Supports baseline search (vector similarity) and hybrid reranker search (vector + keyword).

Returns short answers with citations to the source documents.

Can abstain (say "I don’t know") if evidence is weak.

## How It Works

Step 1: Ingest → break PDFs into chunks and store in SQLite (chunks.db).

Step 2: Index → generate embeddings and build FAISS index.

Step 3: Search → retrieve top results with vector search.

Step 4: Rerank → use hybrid reranker to bring best evidence on top.

Step 5: API → ask questions via FastAPI endpoint /ask.

Step 6: Evaluation → run prepared questions and compare baseline vs hybrid.

## Folder Structure



## Setup Instructions

Clone the repo and go to folder:

git clone <repo-url>
cd RAG-SAFETY-QNA

Create virtual environment and install requirements:

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

Ingest PDFs into chunks:
python src/ingest.py

Build FAISS index:
python src/embed_index.py

Run a quick search:
python src/search.py

Test reranker:
python src/reranker.py

Run the FastAPI server:
python src/api.py


Open http://127.0.0.1:8000/docs
to try /ask.

Evaluate system:
python evaluation.py

Results will be saved in evaluation_output.txt.

Example query:
POST /ask
{
  "q": "What are safety measures in industrial machinery?",
  "k": 4,
  "mode": "hybrid"
}

Example output:
{
  "answer": "The purpose of safety is to protect people and the environment from accidents [1] ...",
  "contexts": [
    { "rank": 1, "doc": "EN_TechnicalguideNo10_REVF.pdf", "score": 1.0, "url": "...", "text": "..." },
    { "rank": 2, "doc": "rep22008e.pdf", "score": 0.72, "url": "...", "text": "..." }
  ],
  "reranker_used": "hybrid",
  "abstained": false
}

### What I Learned

How to build a mini RAG pipeline with ingestion, embeddings, and retrieval using FAISS.

The importance of a reranker (hybrid search with BM25 + vectors) to improve result quality.

How to design extractive answers with citations, keeping outputs grounded in source text.

The value of adding an abstain mechanism when evidence is weak, to keep answers honest.

How to structure a project into clear modules (ingest, embed_index, search, reranker, api, evaluation) for reproducibility.

Writing an evaluation script to compare baseline vs hybrid gave me insight into measuring improvements.

Improved my skills with FastAPI, JSON APIs, and making outputs repeatable with proper seeds and thresholds.


## Thank You

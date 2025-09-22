import os
import sqlite3
from pypdf import PdfReader

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "industrial-safety-pdfs")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chunks.db")

# Created SQLite table for chunks
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_name TEXT,
            chunk_id INTEGER,
            text TEXT
        )
    """)
    conn.commit()
    conn.close()

# Function to split text into chunks (~300 words)
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# Extract text from all PDFs and store chunks
def ingest_pdfs():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for pdf_file in os.listdir(DATA_DIR):
        if not pdf_file.endswith(".pdf"):
            continue
        file_path = os.path.join(DATA_DIR, pdf_file)
        print(f"Processing: {pdf_file}")

        try:
            reader = PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + " "

            chunks = chunk_text(full_text)

            for idx, chunk in enumerate(chunks):
                cur.execute("INSERT INTO chunks (doc_name, chunk_id, text) VALUES (?, ?, ?)",
                            (pdf_file, idx, chunk))

            conn.commit()
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")

    conn.close()
if __name__ == "__main__":
    ingest_pdfs()
    print("✅ All PDFs processed and chunks saved to SQLite.")

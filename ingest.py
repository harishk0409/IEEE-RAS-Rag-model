"""
ingest.py
Reads all .txt/.md files from ./data, chunks them, embeds them,
and stores them in a local ChromaDB vector store (./chroma_db).

Run this once (and again whenever you add/update data files):
    python ingest.py
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "ieee_ras_docs"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between chunks


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window character chunker that tries to break on paragraph/sentence boundaries."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # try to end on a paragraph or sentence boundary if not at the very end
        if end < len(text):
            last_break = max(chunk.rfind("\n\n"), chunk.rfind(". "))
            if last_break > chunk_size * 0.5:  # only trim if it's not too short
                chunk = chunk[:last_break + 1]

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        start += max(len(chunk) - overlap, 1)

    return chunks


def load_documents():
    docs = []
    for filepath in glob.glob(os.path.join(DATA_DIR, "*.*")):
        if filepath.endswith((".txt", ".md")):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            source_name = os.path.basename(filepath)
            for i, chunk in enumerate(chunk_text(text)):
                docs.append({
                    "id": f"{source_name}-{i}",
                    "text": chunk,
                    "source": source_name,
                })
    return docs


def main():
    print("Loading documents from ./data ...")
    docs = load_documents()
    print(f"Found {len(docs)} chunks from files in ./data")

    if not docs:
        print("No documents found! Add .txt or .md files to the data/ folder first.")
        return

    print("Setting up embedding function (all-MiniLM-L6-v2, runs locally)...")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)

    # Fresh collection each time we ingest
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[{"source": d["source"]} for d in docs],
    )

    print(f"Done! Ingested {len(docs)} chunks into '{COLLECTION_NAME}' at ./{DB_DIR}")


if __name__ == "__main__":
    main()

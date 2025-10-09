# code_rag_qdrant.py
from __future__ import annotations
import os, sys, itertools
from pathlib import Path
from typing import Iterable, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# --- 1) Repo -> text chunks ---------------------------------------------------
EXTS = {
    ".py",".js",".ts",".tsx",".jsx",".java",".go",".rs",".cpp",".cc",".c",".h",".cs",
    ".rb",".php",".kt",".scala",".swift",".sql",".sh",".ps1",".html",".css",".scss",
    ".md",".json",".toml",".yaml",".yml",".cmake",".make",".mk",".dockerfile",".ini"
}
SKIP_DIRS = {".git","node_modules","dist","build","out","__pycache__",".venv","venv",".mypy_cache",".idea",".vscode"}

def iter_code_files(root: str) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in EXTS:
                yield p

def chunk_text(text: str, max_chars=1200, overlap=200) -> Iterable[str]:
    step = max(1, max_chars - overlap)
    for i in range(0, len(text), step):
        chunk = text[i:i+max_chars]
        if chunk.strip():
            yield chunk

def yield_repo_chunks(root: str, max_chars=1200, overlap=200) -> Iterable[Dict[str, Any]]:
    for fp in iter_code_files(root):
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        off = 0
        for chunk in chunk_text(txt, max_chars, overlap):
            yield {"path": str(fp), "start": off, "end": off+len(chunk), "text": chunk}
            off += len(chunk) - overlap

# --- 2) Build embeddings + Qdrant (in-memory) --------------------------------
def index_repo_to_qdrant(repo_dir: str, collection="repo_code"):
    # Jina Code embeddings via SentenceTransformers
    # Supports long context; adjust as you like (up to ~8192). :contentReference[oaicite:2]{index=2}
    model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True)
    model.max_seq_length = 2048  # bump if you have more RAM/VRAM

    client = QdrantClient(":memory:")  # local, ephemeral store :contentReference[oaicite:3]{index=3}

    # Create collection with correct vector size
    get_dim = getattr(model, "get_sentence_embedding_dimension", None)
    dim = get_dim() if callable(get_dim) else len(model.encode(["dim-probe"], normalize_embeddings=True)[0])
    client.recreate_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    # Stream chunks -> embeddings -> upsert
    id_counter = itertools.count(1)
    texts, metas = [], []
    BATCH = 64
    for doc in yield_repo_chunks(repo_dir):
        texts.append(doc["text"]); metas.append(doc)
        if len(texts) == BATCH:
            embs = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
            points = [
                models.PointStruct(id=next(id_counter), vector=emb.tolist(), payload=meta)
                for emb, meta in zip(embs, metas)
            ]
            client.upsert(collection_name=collection, points=points)
            texts, metas = [], []
    if texts:
        embs = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        points = [
            models.PointStruct(id=next(id_counter), vector=emb.tolist(), payload=meta)
            for emb, meta in zip(embs, metas)
        ]
        client.upsert(collection_name=collection, points=points)

    return model, client, collection

# --- 3) Simple retrieval ------------------------------------------------------
def search(client: QdrantClient, model: SentenceTransformer, collection: str, query: str, k: int = 5):
    q = model.encode([query], normalize_embeddings=True)[0].tolist()
    hits = client.search(collection_name=collection, query_vector=q, limit=k, with_payload=True)
    for h in hits:
        pl = h.payload
        snippet = (pl.get("text") or "")[:200].replace("\n", " ")
        print(f"{pl['path']}  (score={h.score:.3f})  [{pl['start']}:{pl['end']}]")
        print(f"  {snippet}...\n")

if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "."
    model, client, coll = index_repo_to_qdrant(repo)
    print("Indexed. Try a query (e.g., 'parse json', 'http client', 'vector cosine'):"); q = input("> ")
    search(client, model, coll, q, k=5)

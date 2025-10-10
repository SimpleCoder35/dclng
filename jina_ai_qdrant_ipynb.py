# 1) Install + imports

%pip -q install -U sentence-transformers qdrant-client pandas

from __future__ import annotations
import os, itertools
from pathlib import Path
from typing import Iterable, Dict, Any

import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


# 2) File discovery & chunking (small + readable)
# Which files to index
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
    root = str(Path(root).resolve())
    for fp in iter_code_files(root):
        try:
            txt = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        off = 0
        for ch in chunk_text(txt, max_chars, overlap):
            yield {
                "path": str(Path(fp).relative_to(root)),
                "start": off,
                "end": off + len(ch),
                "text": ch,
            }
            off += len(ch) - overlap


# 3) Index to Qdrant (in‑memory, concise payload)

def index_repo_to_qdrant(
    repo_dir: str,
    collection: str = "repo_code",
    location: str = ":memory:",    # change to "qdrant_db" to persist on disk
    max_seq_len: int = 2048,
    batch_size: int = 64,
) -> tuple[SentenceTransformer, QdrantClient, str]:
    # Load Jina code embeddings through SentenceTransformers
    model = SentenceTransformer("jinaai/jina-embeddings-v2-base-code", trust_remote_code=True)
    model.max_seq_length = max_seq_len

    # Embedded Qdrant; ":memory:" is ephemeral. Use a folder path to persist.
    client = QdrantClient(location=location)

    # Create collection with correct dimension
    dim = model.get_sentence_embedding_dimension()
    client.recreate_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )

    # Stream chunks -> embeddings -> upsert
    texts, metas = [], []
    id_counter = itertools.count(1)

    for doc in yield_repo_chunks(repo_dir):
        # Keep payload concise: no full text, just a short preview + coordinates
        meta = {
            "path": doc["path"],
            "start": doc["start"],
            "end": doc["end"],
            "preview": doc["text"][:400],
        }
        texts.append(doc["text"])
        metas.append(meta)

        if len(texts) >= batch_size:
            embs = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
            points = [
                models.PointStruct(id=next(id_counter), vector=emb.tolist(), payload=m)
                for emb, m in zip(embs, metas)
            ]
            client.upsert(collection_name=collection, points=points)
            texts.clear(); metas.clear()

    if texts:
        embs = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        points = [
            models.PointStruct(id=next(id_counter), vector=emb.tolist(), payload=m)
            for emb, m in zip(embs, metas)
        ]
        client.upsert(collection_name=collection, points=points)

    return model, client, collection

# 4) Simple retrieval → DataFrame (great for Jupyter)
def search(client: QdrantClient, model: SentenceTransformer, collection: str, query: str, k: int = 5) -> pd.DataFrame:
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()
    hits = client.search(collection_name=collection, query_vector=qvec, limit=k, with_payload=True)
    rows = []
    for h in hits:
        pl = h.payload or {}
        rows.append({
            "score": round(h.score, 3),
            "path": pl.get("path", ""),
            "span": f"{pl.get('start', 0)}:{pl.get('end', 0)}",
            "preview": (pl.get("preview", "") or "").replace("\n", " "),
        })
    return pd.DataFrame(rows, columns=["score","path","span","preview"])

# 5) Run it

# Choose the repo to index
REPO_DIR = "."  # or "/absolute/path/to/your/repo"

model, client, coll = index_repo_to_qdrant(REPO_DIR)
display(search(client, model, coll, query="parse json config", k=5))

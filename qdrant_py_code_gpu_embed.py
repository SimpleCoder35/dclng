# #!/usr/bin/env python3
# # code_indexer.py — Python/ipynb code indexer with AST-aware chunking + embedded Qdrant

# from __future__ import annotations
# import os, ast, json, sys
# from pathlib import Path
# from typing import Iterable, Dict, Any, List, Tuple

# from qdrant_client import QdrantClient, models as qm
# from sentence_transformers import SentenceTransformer

# # ---------- discovery ----------
# EXTS = {".py", ".ipynb"}
# SKIP = {"node_modules",".git","build","dist","out","__pycache__",".venv","venv",
#         ".mypy_cache",".pytest_cache",".ipynb_checkpoints"}

# def iter_code_files(root: str|Path) -> Iterable[Path]:
#     root = Path(root).resolve()
#     for dpath, dnames, fnames in os.walk(root):
#         dnames[:] = [d for d in dnames if d not in SKIP]
#         for fn in fnames:
#             p = Path(dpath, fn)
#             if p.suffix.lower() in EXTS:
#                 yield p

# # ---------- read .py + .ipynb (code only) ----------
# def extract_code_units(path: Path) -> List[Tuple[str, str]]:
#     if path.suffix.lower() == ".py":
#         return [(str(path), path.read_text("utf-8", errors="ignore"))]
#     nb = json.loads(path.read_text("utf-8", errors="ignore"))
#     out = []
#     for i, c in enumerate(nb.get("cells", []), 1):
#         if c.get("cell_type") == "code":
#             src = "".join(c.get("source", []))
#             # strip IPython magics/shell lines, keep code + comments
#             lines = [ln for ln in src.splitlines()
#                      if not ln.lstrip().startswith(("%","%%","!","?"))]
#             out.append((f"{path}#cell-{i}", "\n".join(lines)))
#     return out

# # ---------- python-aware chunking ----------
# def _leading_comment_start(lines: List[str], start_line: int, max_up: int = 5) -> int:
#     i, seen = start_line - 2, 0
#     while i >= 0 and seen < max_up:
#         s = lines[i].strip()
#         if s.startswith("#"):
#             seen += 1; i -= 1
#         elif s == "":
#             i -= 1
#         else:
#             break
#     return i + 2

# def _slice(lines: List[str], a: int, b: int) -> str:
#     return "\n".join(lines[a-1:b])

# def _split_long(text: str, max_chars: int, overlap: int):
#     step = max(1, max_chars - overlap)
#     for off in range(0, len(text), step):
#         ch = text[off:off+max_chars]
#         if ch.strip():
#             yield off, off+len(ch), ch

# def chunk_python_text(path: str, text: str, max_chars=1600, overlap=200) -> List[Dict[str, Any]]:
#     lines = text.splitlines()
#     if not text.strip():
#         return []
#     try:
#         tree = ast.parse(text)
#         items: List[Tuple[str,str,int,int]] = []
#         for node in tree.body:
#             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
#                 items.append(("func", node.name, node.lineno, getattr(node, "end_lineno", node.lineno)))
#             elif isinstance(node, ast.ClassDef):
#                 for n in node.body:
#                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
#                         items.append(("method", f"{node.name}.{n.name}",
#                                       n.lineno, getattr(n, "end_lineno", n.lineno)))
#         # uncovered top-level chunks (imports / module code)
#         covered = [False]*(len(lines)+1)
#         for _,_,a,b in items:
#             for i in range(a, b+1): covered[i] = True
#         out: List[Dict[str,Any]] = []
#         a = None
#         for i in range(1, len(lines)+1):
#             if not covered[i] and lines[i-1].strip() and a is None:
#                 a = i
#             if (i == len(lines) or covered[i]) and a is not None:
#                 b = i-1 if covered[i] else i
#                 out.append({"type":"module","name":"module","path":path,
#                             "start_line":a,"end_line":b,"text":_slice(lines,a,b)})
#                 a = None
#         for typ,name,a,b in items:
#             a2 = _leading_comment_start(lines, a)
#             out.append({"type":typ,"name":name,"path":path,
#                         "start_line":a2,"end_line":b,"text":_slice(lines,a2,b)})
#     except Exception:
#         out = [{"type":"window","name":"window","path":path,
#                 "start_line":1,"end_line":len(lines),"text":text}]

#     final = []
#     for c in out:
#         t = c["text"]
#         if len(t) > max_chars:
#             for i,(s,e,sub) in enumerate(_split_long(t, max_chars, overlap),1):
#                 d = c.copy(); d.update({"text":sub,"part":i,"char_start":s,"char_end":e})
#                 final.append(d)
#         else:
#             final.append(c)
#     return final

# def stream_python_chunks(root: str|Path, max_chars=1600, overlap=200) -> Iterable[Dict[str, Any]]:
#     for p in iter_code_files(root):
#         for vpath, code in extract_code_units(p):
#             for ch in chunk_python_text(vpath, code, max_chars, overlap):
#                 ch["id"] = f"{ch['path']}#{ch.get('name','')}-{ch.get('start_line',1)}-{ch.get('end_line',1)}"
#                 ch["preview"] = ch["text"][:240].replace("\n"," ")
#                 yield ch

# # ---------- index (embedded Qdrant, in-memory) ----------
# def index_repo(
#     repo_dir: str|Path,
#     collection: str = "repo_code",
#     model_name: str = "qwen/qwen3-embedding-6.8b",
#     location: str = ":memory:",
#     max_seq_len: int = 8192,
#     batch_size: int = 128,
#     encode_batch_size: int = 64
# ) -> dict:
#     import importlib
#     torch = importlib.import_module("torch")
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
#     model.max_seq_length = max_seq_len
#     dim = model.get_sentence_embedding_dimension()

#     client = QdrantClient(location=location)
#     try: client.delete_collection(collection)
#     except Exception: pass
#     client.create_collection(
#         collection_name=collection,
#         vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
#     )

#     ids, texts, payloads, total = [], [], [], 0
#     for doc in stream_python_chunks(repo_dir):
#         ids.append(doc["id"]); texts.append(doc["text"])
#         payloads.append({k: doc[k] for k in ("path","type","name","start_line","end_line","preview")})
#         if len(texts) >= batch_size:
#             vecs = model.encode(texts, batch_size=encode_batch_size,
#                                 normalize_embeddings=True, show_progress_bar=False)
#             client.upsert(collection_name=collection,
#                           points=qm.Batch(ids=ids, vectors=vecs.tolist(), payloads=payloads))
#             total += len(texts); ids.clear(); texts.clear(); payloads.clear()
#     if texts:
#         vecs = model.encode(texts, batch_size=encode_batch_size,
#                             normalize_embeddings=True, show_progress_bar=False)
#         client.upsert(collection_name=collection,
#                       points=qm.Batch(ids=ids, vectors=vecs.tolist(), payloads=payloads))
#         total += len(texts)

#     return {"client": client, "model": model, "collection": collection,
#             "device": device, "points": total}

# # ---------- retrieval ----------
# def search(index: dict, query: str, k: int = 5) -> Dict[str, Any]:
#     model: SentenceTransformer = index["model"]
#     client: QdrantClient = index["client"]
#     collection = index["collection"]

#     qvec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
#     try:
#         resp = client.query_points(collection_name=collection,
#                                    query=qm.NearVector(vector=qvec),
#                                    limit=k, with_payload=True)
#         hits = resp.points
#     except Exception:
#         hits = client.search(collection_name=collection,
#                              query_vector=qvec, limit=k, with_payload=True)
#     results = []
#     for h in hits:
#         pl = h.payload or {}
#         results.append({
#             "id": str(h.id), "score": float(h.score), "path": pl.get("path"),
#             "name": pl.get("name"), "type": pl.get("type"),
#             "span": f"{pl.get('start_line')}:{pl.get('end_line')}",
#             "preview": pl.get("preview","")
#         })
#     return {"query": query, "k": k, "count": len(results),
#             "collection": collection, "device": index["device"],
#             "results": results}

# # ---------- CLI (optional) ----------
# # if __name__ == "__main__":
# #     repo = sys.argv[1] if len(sys.argv) > 1 else "."
# #     idx = index_repo(repo)
# #     print(f"Indexed {repo} -> {idx['points']} chunks | device={idx['device']}")
# #     try:
# #         while True:
# #             q = input("🔎 query (blank to exit): ").strip()
# #             if not q: break
# #             resp = search(idx, q, k=5)
# #             for r in resp["results"]:
# #                 print(f"{r['score']:.3f}  {r['path']}  {r['span']}  {r['name']}  {r['preview'][:100]}...")
# #     except KeyboardInterrupt:
# #         pass


# # paste the script in a cell or save as code_indexer.py and import
# idx = index_repo("/path/to/repo")        # in-memory, GPU if available
# res = search(idx, "fit() early stopping", k=5)
# res["results"][:3]


#######################################################################################################################
#Fixed UUID Issue:
#######################################################################################################################

# python_code_indexer.py — Python/ipynb AST-aware indexer + embedded Qdrant (UUID ids)

from __future__ import annotations
import os, ast, json, uuid
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

from qdrant_client import QdrantClient, models as qm
from sentence_transformers import SentenceTransformer

EXTS = {".py", ".ipynb"}
SKIP = {"node_modules",".git","build","dist","out","__pycache__",".venv","venv",
        ".mypy_cache",".pytest_cache",".ipynb_checkpoints"}

# ---------- discovery ----------
def iter_code_files(root: str|Path) -> Iterable[Path]:
    root = Path(root).resolve()
    for dpath, dnames, fnames in os.walk(root):
        dnames[:] = [d for d in dnames if d not in SKIP]
        for fn in fnames:
            p = Path(dpath, fn)
            if p.suffix.lower() in EXTS:
                yield p

# ---------- read .py + .ipynb (code/comments only) ----------
def extract_code_units(path: Path) -> List[Tuple[str, str]]:
    if path.suffix.lower() == ".py":
        return [(str(path), path.read_text("utf-8", errors="ignore"))]
    nb = json.loads(path.read_text("utf-8", errors="ignore"))
    out = []
    for i, c in enumerate(nb.get("cells", []), 1):
        if c.get("cell_type") == "code":
            src = "".join(c.get("source", []))
            lines = [ln for ln in src.splitlines()
                     if not ln.lstrip().startswith(("%","%%","!","?"))]
            out.append((f"{path}#cell-{i}", "\n".join(lines)))
    return out

# ---------- python-aware chunking ----------
def _leading_comment_start(lines: List[str], start_line: int, max_up: int = 5) -> int:
    i, seen = start_line - 2, 0
    while i >= 0 and seen < max_up:
        s = lines[i].strip()
        if s.startswith("#"): seen += 1; i -= 1
        elif s == "": i -= 1
        else: break
    return i + 2

def _slice(lines: List[str], a: int, b: int) -> str:
    return "\n".join(lines[a-1:b])

def _split_long(text: str, max_chars: int, overlap: int):
    step = max(1, max_chars - overlap)
    for off in range(0, len(text), step):
        ch = text[off:off+max_chars]
        if ch.strip(): yield off, off+len(ch), ch

def chunk_python_text(path: str, text: str, max_chars=1600, overlap=200) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    if not text.strip(): return []
    try:
        tree = ast.parse(text)
        items: List[Tuple[str,str,int,int]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                items.append(("func", node.name, node.lineno, getattr(node,"end_lineno",node.lineno)))
            elif isinstance(node, ast.ClassDef):
                for n in node.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        items.append(("method", f"{node.name}.{n.name}",
                                      n.lineno, getattr(n,"end_lineno",n.lineno)))
        covered = [False]*(len(lines)+1)
        for _,_,a,b in items:
            for i in range(a, b+1): covered[i] = True
        out: List[Dict[str,Any]] = []
        a = None
        for i in range(1, len(lines)+1):
            if not covered[i] and lines[i-1].strip() and a is None: a = i
            if (i == len(lines) or covered[i]) and a is not None:
                b = i-1 if covered[i] else i
                out.append({"type":"module","name":"module","path":path,
                            "start_line":a,"end_line":b,"text":_slice(lines,a,b)})
                a = None
        for typ,name,a,b in items:
            a2 = _leading_comment_start(lines, a)
            out.append({"type":typ,"name":name,"path":path,
                        "start_line":a2,"end_line":b,"text":_slice(lines,a2,b)})
    except Exception:
        out = [{"type":"window","name":"window","path":path,
                "start_line":1,"end_line":len(lines),"text":text}]

    final = []
    for c in out:
        t = c["text"]
        if len(t) > max_chars:
            for i,(s,e,sub) in enumerate(_split_long(t, max_chars, overlap),1):
                d = c.copy(); d.update({"text":sub,"part":i,"char_start":s,"char_end":e})
                final.append(d)
        else:
            final.append(c)
    return final

def stream_python_chunks(root: str|Path, max_chars=1600, overlap=200) -> Iterable[Dict[str, Any]]:
    for p in iter_code_files(root):
        for vpath, code in extract_code_units(p):
            for ch in chunk_python_text(vpath, code, max_chars, overlap):
                # Deterministic UUIDv5 id (Qdrant requires int or UUID)
                sid = f"{ch['path']}|{ch.get('name','')}|{ch.get('start_line',1)}|{ch.get('end_line',1)}|{ch.get('part',0)}"
                ch["id"] = uuid.uuid5(uuid.NAMESPACE_URL, sid)
                ch["preview"] = ch["text"][:240].replace("\n"," ")
                yield ch

# ---------- index (embedded Qdrant, in-memory) ----------
def index_repo(
    repo_dir: str|Path,
    collection: str = "repo_code",
    model_name: str = "qwen/qwen3-embedding-6.8b",
    location: str = ":memory:",
    max_seq_len: int = 8192,
    batch_size: int = 128,
    encode_batch_size: int = 64
) -> dict:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    model.max_seq_length = max_seq_len
    dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(location=location)
    try: client.delete_collection(collection)
    except Exception: pass
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )

    ids, texts, payloads, total = [], [], [], 0
    for doc in stream_python_chunks(repo_dir):
        ids.append(doc["id"])  # UUID object
        texts.append(doc["text"])
        payloads.append({k: doc[k] for k in ("path","type","name","start_line","end_line","preview")})
        if len(texts) >= batch_size:
            vecs = model.encode(texts, batch_size=encode_batch_size,
                                normalize_embeddings=True, show_progress_bar=False)
            client.upsert(collection_name=collection,
                          points=qm.Batch(ids=ids, vectors=vecs.tolist(), payloads=payloads))
            total += len(texts); ids.clear(); texts.clear(); payloads.clear()
    if texts:
        vecs = model.encode(texts, batch_size=encode_batch_size,
                            normalize_embeddings=True, show_progress_bar=False)
        client.upsert(collection_name=collection,
                      points=qm.Batch(ids=ids, vectors=vecs.tolist(), payloads=payloads))
        total += len(texts)

    return {"client": client, "model": model, "collection": collection,
            "device": device, "points": total}

# ---------- retrieval ----------
def search(index: dict, query: str, k: int = 5) -> Dict[str, Any]:
    model: SentenceTransformer = index["model"]
    client: QdrantClient = index["client"]
    collection = index["collection"]

    qvec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
    try:
        resp = client.query_points(collection_name=collection,
                                   query=qm.NearVector(vector=qvec),
                                   limit=k, with_payload=True)
        hits = resp.points
    except Exception:
        hits = client.search(collection_name=collection,
                             query_vector=qvec, limit=k, with_payload=True)
    results = []
    for h in hits:
        pl = h.payload or {}
        results.append({
            "id": str(h.id), "score": float(h.score), "path": pl.get("path"),
            "name": pl.get("name"), "type": pl.get("type"),
            "span": f"{pl.get('start_line')}:{pl.get('end_line')}",
            "preview": pl.get("preview","")
        })
    return {"query": query, "k": k, "count": len(results),
            "collection": collection, "device": index["device"], "results": results}

################
idx = index_repo("/path/to/codebase")   # in-memory, GPU if available
resp = search(idx, "train_test_split stratify", k=5)
resp["results"][:3]


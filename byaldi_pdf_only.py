# pip install byaldi

from pathlib import Path
from typing import List, Dict, Any
from byaldi import RAGMultiModalModel

MODEL_ID = "vidore/colqwen2.5-v0.2"   # or "vidore/colpali"
INDEX_NAME = "gov_pdf_byaldi"

def build_pdf_index(pdf_root: str, index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    """
    Index all PDFs under pdf_root. Stores page images inside the index.
    """
    root = str(Path(pdf_root).expanduser().resolve())
    rag = RAGMultiModalModel.from_pretrained(MODEL_ID)
    rag.index(
        input_path=root,
        index_name=index_name,
        store_collection_with_index=True,   # keep page images for b64 returns
        overwrite=True
    )
    return rag

def load_pdf_index(index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    """
    Reload an existing index (after a kernel restart, etc.).
    """
    rag = RAGMultiModalModel.from_pretrained(MODEL_ID)
    rag.load_index(index_name=index_name)
    return rag

def _norm(hit: Dict[str, Any]) -> Dict[str, Any]:
    score = hit.get("score", hit.get("_score"))
    page  = int(hit.get("page_num", hit.get("page", hit.get("page_idx", 1))))
    src   = hit.get("source") or hit.get("document") or hit.get("file_path") or hit.get("path")
    b64   = hit.get("image_base64") or hit.get("b64_image") or hit.get("image_b64")
    return {
        "doc": Path(src).name if src else None,
        "path": src,
        "page": page,
        "score": float(score) if score is not None else None,
        "image_base64": b64,   # ready for VLMs or embedding as data URI
        "raw": hit
    }

def search_images(rag: RAGMultiModalModel, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Return top-k hits with base64 page images and doc/page metadata.
    """
    hits = rag.search(query, k=k)
    return [_norm(h) for h in hits]

# --- Example usage ---
# rag = build_pdf_index("/path/to/folder_with_pdfs")
# # (later) rag = load_pdf_index()
# results = search_images(rag, "validation ROC curve", k=5)
# for r in results:
#     print(r["doc"], f"p{r['page']}", r["score"])
# # r["image_base64"] -> 'data:image/png;base64,...'


#### v2( earlier getting error)

# pip install byaldi

from pathlib import Path
from typing import List, Dict, Any
from byaldi import RAGMultiModalModel

MODEL_ID = "vidore/colqwen2.5-v0.2"   # or "vidore/colpali"
INDEX_NAME = "gov_pdf_byaldi"

def _list_pdfs(pdf_root: str, ignore_hidden: bool = True) -> List[str]:
    """
    Return a sorted, deduped list of absolute PDF file paths.
    - Only *.pdf files
    - Skips any path containing a hidden (dot) directory
    - Skips macOS resource-fork files like '._file.pdf'
    """
    root = Path(pdf_root).expanduser().resolve()
    seen = set()
    out: List[str] = []

    for p in root.rglob("*.pdf"):
        if not p.is_file():
            continue
        # Skip dot-directories anywhere in the relative path
        if ignore_hidden:
            rel_parts = p.relative_to(root).parts
            if any(part.startswith(".") for part in rel_parts):
                continue
        # Skip Apple resource fork artifacts
        if p.name.startswith("._"):
            continue
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(rp)

    return sorted(out)

def build_pdf_index(pdf_root: str, index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    """
    Index only the cleaned list of PDFs. Stores page images inside the index.
    Passing an explicit file list prevents doc_ids/input_items mismatch.
    """
    pdf_files = _list_pdfs(pdf_root)
    if not pdf_files:
        raise ValueError(f"No PDFs found under: {pdf_root}")

    print(f"[INFO] Indexing {len(pdf_files)} PDFs...")
    rag = RAGMultiModalModel.from_pretrained(MODEL_ID)
    rag.index(
        input_path=pdf_files,                  # explicit list -> PDFs only
        index_name=index_name,
        store_collection_with_index=True,      # keep page images for b64 returns
        overwrite=True
    )
    print(f"[OK] Index built: {index_name}")
    return rag

def load_pdf_index(index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    """Reload an existing index (after a kernel restart, etc.)."""
    rag = RAGMultiModalModel.from_pretrained(MODEL_ID)
    rag.load_index(index_name=index_name)
    return rag

def _norm(hit: Dict[str, Any]) -> Dict[str, Any]:
    score = hit.get("score", hit.get("_score"))
    page  = int(hit.get("page_num", hit.get("page", hit.get("page_idx", 1))))
    src   = hit.get("source") or hit.get("document") or hit.get("file_path") or hit.get("path")
    b64   = hit.get("image_base64") or hit.get("b64_image") or hit.get("image_b64")
    return {
        "doc": Path(src).name if src else None,
        "path": src,
        "page": page,
        "score": float(score) if score is not None else None,
        "image_base64": b64,   # ready for VLMs or embedding as data URI
        "raw": hit
    }

def search_images(rag: RAGMultiModalModel, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Return top-k hits with base64 page images and doc/page metadata."""
    hits = rag.search(query, k=k)
    return [_norm(h) for h in hits]

# --- Example usage ---
# rag = build_pdf_index("/path/to/folder_with_pdfs")
# # (later) rag = load_pdf_index()
# results = search_images(rag, "validation ROC curve", k=5)
# for r in results:
#     print(r["doc"], f"p{r['page']}", r["score"])
# # r["image_base64"] -> base64 PNG for the page

## v3(earlier verasion has issue wiht shoudl be path like or file, but list given issue)

# pip install byaldi

import os, shutil
from pathlib import Path
from typing import List, Dict, Any
from byaldi import RAGMultiModalModel

MODEL_ID   = "vidore/colqwen2.5-v0.2"   # or "vidore/colpali"
INDEX_NAME = "gov_pdf_byaldi"
STAGE_ROOT = ".byaldi_stage"            # temp folder to hold PDFs-only view

def _list_pdfs(pdf_root: str, ignore_hidden=True) -> List[Path]:
    root = Path(pdf_root).expanduser().resolve()
    pdfs: List[Path] = []
    for p in root.rglob("*.pdf"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if ignore_hidden and any(part.startswith(".") for part in rel_parts):
            continue
        if p.name.startswith("._"):  # macOS resource forks
            continue
        pdfs.append(p.resolve())
    # de-dupe & stable sort
    seen, out = set(), []
    for p in sorted(pdfs):
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def _stage_pdfs_only(pdf_root: str, stage_dir: Path) -> Path:
    """Create a mirror dir with *only* PDFs (no dot-dirs)."""
    root = Path(pdf_root).expanduser().resolve()
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    pdfs = _list_pdfs(pdf_root)
    if not pdfs:
        raise ValueError(f"No PDFs found under: {root}")

    for src in pdfs:
        rel = src.parent.relative_to(root)  # preserve structure (avoids filename collisions)
        dst_dir = stage_dir / rel
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)  # simple & robust; swap to hardlink if you prefer
    return stage_dir

def build_pdf_index(pdf_root: str, index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    """Stage PDFs into a clean directory, then index that *directory* (path-like)."""
    stage_dir = Path(STAGE_ROOT) / index_name
    staged = _stage_pdfs_only(pdf_root, stage_dir)

    print(f"[INFO] Building index from staged dir: {staged}")
    rag = RAGMultiModalModel.from_pretrained(MODEL_ID)
    rag.index(
        input_path=str(staged),              # <-- pass a *directory path*, not a list
        index_name=index_name,
        store_collection_with_index=True,    # embed base64 page images in index
        overwrite=True
    )
    print(f"[OK] Index built: {index_name}")
    return rag

def append_pdfs_to_index(pdf_root: str, index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    """Append new PDFs to an existing index using add_to_index()."""
    stage_dir = Path(STAGE_ROOT) / f"{index_name}_append"
    staged = _stage_pdfs_only(pdf_root, stage_dir)

    rag = RAGMultiModalModel.from_index(index_name)  # load existing
    # You can pass a dir here; Byaldi will ingest its PDFs.
    rag.add_to_index(str(staged), store_collection_with_index=True)
    print(f"[OK] Appended PDFs from {staged} into index: {index_name}")
    return rag

def load_pdf_index(index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    return RAGMultiModalModel.from_index(index_name)

def _norm(hit: Dict[str, Any]) -> Dict[str, Any]:
    score = hit.get("score", hit.get("_score"))
    page  = int(hit.get("page_num", hit.get("page", hit.get("page_idx", 1))))
    src   = hit.get("source") or hit.get("document") or hit.get("file_path") or hit.get("path")
    b64   = hit.get("image_base64") or hit.get("b64_image") or hit.get("image_b64")
    return {
        "doc": Path(src).name if src else None,
        "path": src,
        "page": page,
        "score": float(score) if score is not None else None,
        "image_base64": b64,
        "raw": hit
    }

def search_images(rag: RAGMultiModalModel, query: str, k: int = 5):
    return [_norm(h) for h in rag.search(query, k=k)]

# --- Example ---
# rag = build_pdf_index("/path/to/folder_with_pdfs")
# results = search_images(rag, "validation ROC curve", k=5)
# for r in results: print(r["doc"], f"p{r['page']}", r["score"])

## v4(previos has path like issue.)

# pip install byaldi

import os, shutil
from pathlib import Path
from typing import List, Dict, Any
from byaldi import RAGMultiModalModel

MODEL_ID   = "vidore/colqwen2.5-v0.2"   # or "vidore/colpali"
INDEX_NAME = "gov_pdf_byaldi"
STAGE_ROOT = ".byaldi_stage"            # temp folder to hold PDFs-only view

def _list_pdfs(pdf_root: str, ignore_hidden=True) -> List[Path]:
    root = Path(pdf_root).expanduser().resolve()
    pdfs: List[Path] = []
    for p in root.rglob("*.pdf"):
        if not p.is_file():
            continue
        rel_parts = p.relative_to(root).parts
        if ignore_hidden and any(part.startswith(".") for part in rel_parts):
            continue
        if p.name.startswith("._"):  # macOS resource forks
            continue
        pdfs.append(p.resolve())
    # de-dupe & stable sort
    seen, out = set(), []
    for p in sorted(pdfs):
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def _stage_pdfs_only(pdf_root: str, stage_dir: Path) -> Path:
    """Create a mirror dir with *only* PDFs (no dot-dirs)."""
    root = Path(pdf_root).expanduser().resolve()
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    pdfs = _list_pdfs(pdf_root)
    if not pdfs:
        raise ValueError(f"No PDFs found under: {root}")

    for src in pdfs:
        rel = src.parent.relative_to(root)  # preserve structure (avoids filename collisions)
        dst_dir = stage_dir / rel
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)  # simple & robust; swap to hardlink if you prefer
    return stage_dir

def build_pdf_index(pdf_root: str, index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    """Stage PDFs into a clean directory, then index that *directory* (path-like)."""
    stage_dir = Path(STAGE_ROOT) / index_name
    staged = _stage_pdfs_only(pdf_root, stage_dir)

    print(f"[INFO] Building index from staged dir: {staged}")
    rag = RAGMultiModalModel.from_pretrained(MODEL_ID)
    rag.index(
        input_path=str(staged),              # <-- pass a *directory path*, not a list
        index_name=index_name,
        store_collection_with_index=True,    # embed base64 page images in index
        overwrite=True
    )
    print(f"[OK] Index built: {index_name}")
    return rag

def append_pdfs_to_index(pdf_root: str, index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    """Append new PDFs to an existing index using add_to_index()."""
    stage_dir = Path(STAGE_ROOT) / f"{index_name}_append"
    staged = _stage_pdfs_only(pdf_root, stage_dir)

    rag = RAGMultiModalModel.from_index(index_name)  # load existing
    # You can pass a dir here; Byaldi will ingest its PDFs.
    rag.add_to_index(str(staged), store_collection_with_index=True)
    print(f"[OK] Appended PDFs from {staged} into index: {index_name}")
    return rag

def load_pdf_index(index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    return RAGMultiModalModel.from_index(index_name)

def _norm(hit: Dict[str, Any]) -> Dict[str, Any]:
    score = hit.get("score", hit.get("_score"))
    page  = int(hit.get("page_num", hit.get("page", hit.get("page_idx", 1))))
    src   = hit.get("source") or hit.get("document") or hit.get("file_path") or hit.get("path")
    b64   = hit.get("image_base64") or hit.get("b64_image") or hit.get("image_b64")
    return {
        "doc": Path(src).name if src else None,
        "path": src,
        "page": page,
        "score": float(score) if score is not None else None,
        "image_base64": b64,
        "raw": hit
    }

def search_images(rag: RAGMultiModalModel, query: str, k: int = 5):
    return [_norm(h) for h in rag.search(query, k=k)]

# --- Example ---
# rag = build_pdf_index("/path/to/folder_with_pdfs")
# results = search_images(rag, "validation ROC curve", k=5)
# for r in results: print(r["doc"], f"p{r['page']}", r["score"])


##############################################################################################################

# pip install byaldi

from pathlib import Path
from typing import List, Dict, Any, Iterable
import shutil
from byaldi import RAGMultiModalModel

MODEL_ID   = "vidore/colqwen2.5-v0.2"      # or "vidore/colpali"
INDEX_NAME = "gov_pdf_byaldi"
STAGE_DIR  = Path(".byaldi_stage") / INDEX_NAME  # where we stage only PDFs

# --- helpers ---
def _iter_pdfs(root: str, ignore_hidden: bool = True) -> Iterable[Path]:
    root = Path(root).expanduser().resolve()
    for p in root.rglob("*.pdf"):
        if not p.is_file(): 
            continue
        rel_parts = p.relative_to(root).parts
        if ignore_hidden and any(part.startswith(".") for part in rel_parts):
            continue
        if p.name.startswith("._"):  # macOS resource-fork junk
            continue
        yield p.resolve()

def _stage_pdfs_only(src_root: str, dst_dir: Path) -> Path:
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    root = Path(src_root).expanduser().resolve()

    count = 0
    for src in sorted(set(_iter_pdfs(root))):
        rel = src.parent.relative_to(root)
        out_dir = dst_dir / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        count += 1
    if count == 0:
        raise ValueError(f"No PDFs found under: {root}")
    return dst_dir

def _asdict(res: Any) -> Dict[str, Any]:
    if isinstance(res, dict): 
        return res
    for m in ("model_dump", "dict"):
        if hasattr(res, m):
            try:
                d = getattr(res, m)()
                if isinstance(d, dict): 
                    return d
            except TypeError:
                pass
    d = {}
    for k in ("score","page_num","base64","image_base64","b64_image","source","path","file_path","document"):
        if hasattr(res, k): d[k] = getattr(res, k)
    try: d.update(dict(res))
    except Exception: pass
    return d

def _norm(res: Any) -> Dict[str, Any]:
    r = _asdict(res)
    score = r.get("score")
    page  = int(r.get("page_num", 1))
    path  = r.get("path") or r.get("source") or r.get("file_path") or r.get("document")
    b64   = r.get("base64") or r.get("image_base64") or r.get("b64_image")
    return {
        "filename": Path(path).name if path else None,
        "path": path,
        "page": page,
        "score": float(score) if score is not None else None,
        "image_base64": b64,
        "raw": r,
    }

# --- core api ---
def build_pdf_index(pdf_root: str, index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    staged = _stage_pdfs_only(pdf_root, STAGE_DIR)
    rag = RAGMultiModalModel.from_pretrained(MODEL_ID)
    rag.index(
        input_path=str(staged),                 # directory (path-like), not a list
        index_name=index_name,
        store_collection_with_index=True,       # embeds page images for b64 in results
        overwrite=True,
    )
    return rag

def load_pdf_index(index_name: str = INDEX_NAME) -> RAGMultiModalModel:
    return RAGMultiModalModel.from_index(index_name)

def search_images(rag: RAGMultiModalModel, query: str, k: int = 5) -> List[Dict[str, Any]]:
    return [_norm(h) for h in rag.search(query, k=k)]

# --- example usage ---
# rag = build_pdf_index("/path/to/folder_with_pdfs")
# # (later) rag = load_pdf_index()
# hits = search_images(rag, "validation ROC curve", k=5)
# for h in hits:
#     print(h["filename"], f"p{h['page']}", h["score"], "img?" if bool(h["image_base64"]) else "no-img")


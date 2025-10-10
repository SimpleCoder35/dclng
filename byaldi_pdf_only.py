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

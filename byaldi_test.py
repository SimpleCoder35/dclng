from pathlib import Path
from byaldi import RAGMultiModalModel

MODEL = "vidore/colqwen2.5-v0.2"
PDF_DIR = Path("pdfs")
INDEX_NAME = "colqwen25_pdf_index"

def main():
    # load model
    rag = RAGMultiModalModel.from_pretrained(MODEL)

    # build index over PDFs (Byaldi uses pdf2image → Poppler under the hood)
    rag.index(
        input_path=str(PDF_DIR),   # file OR directory
        index_name=INDEX_NAME,
        overwrite=True
    )

    # query
    for q in ["total revenue", "R&D headcount", "operating margin"]:
        hits = rag.search(q, k=3)
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  -> score={h['score']:.3f}  doc={h['doc_id']}  page={h['page_num']}")

    # (optional) reload later without re-encoding
    rag2 = RAGMultiModalModel.from_index(INDEX_NAME)
    print("\nReload check:", rag2.search("executive summary", k=1))

if __name__ == "__main__":
    main()

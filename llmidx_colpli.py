import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import fitz
import requests
import torch
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from transformers.utils import is_flash_attn_2_available


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DEFAULT_PDF_URL = "https://arxiv.org/pdf/2402.05120"
DEFAULT_MODEL_ID = "vidore/colqwen2.5-v0.2"
DEFAULT_QUESTION = "lama's 70b gsm8ks accuracy measures over different ensamble sizes"


@dataclass
class PageRecord:
    page_number: int
    text: str
    image_path: Path


def download_pdf(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        logging.info("PDF already present at %s", destination)
        return

    logging.info("Downloading PDF from %s", url)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    logging.info("Saved PDF to %s", destination)


def render_pdf_to_images(pdf_path: Path, output_dir: Path, refresh: bool = False) -> List[PageRecord]:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Rendering %s into page images", pdf_path)
    records: List[PageRecord] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page_index, page in enumerate(doc, start=1):
            image_path = output_dir / f"page_{page_index:03d}.png"
            if refresh or not image_path.exists():
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                pix.save(str(image_path))
            text = page.get_text()
            records.append(PageRecord(page_number=page_index, text=text, image_path=image_path))
    finally:
        doc.close()
    logging.info("Prepared %d page records", len(records))
    return records


def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device.index or 0)
        if capability[0] >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def move_to_device(batch, device: torch.device, dtype: torch.dtype):
    prepared = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
                prepared[key] = value.to(device=device, dtype=dtype)
            else:
                prepared[key] = value.to(device=device)
        else:
            prepared[key] = value
    return prepared


def load_colqwen(model_id: str, device: torch.device):
    dtype = select_dtype(device)
    attn_impl = None
    if device.type == "cuda" and is_flash_attn_2_available():
        attn_impl = "flash_attention_2"
    logging.info("Loading %s on %s (dtype=%s)", model_id, device, dtype)
    model = ColQwen2_5.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device.type,
        attn_implementation=attn_impl,
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(model_id)
    return model, processor, dtype


def compute_page_embeddings(
    pages: Iterable[PageRecord],
    model: ColQwen2_5,
    processor: ColQwen2_5_Processor,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    embeddings: List[torch.Tensor] = []
    for page in pages:
        with Image.open(page.image_path) as image_file:
            pil_image = image_file.convert("RGB")
            batch_images = processor.process_images([pil_image])
        batch_images = move_to_device(batch_images, device, dtype)
        with torch.no_grad():
            image_embedding = model(**batch_images)
        mask = batch_images["attention_mask"][0].bool()
        embeddings.append(image_embedding[0][mask].detach().cpu())
    return embeddings


def build_nodes(pages: Iterable[PageRecord], source_pdf: Path) -> List[TextNode]:
    nodes: List[TextNode] = []
    pdf_name = source_pdf.name
    for page in pages:
        metadata = {
            "page_number": page.page_number,
            "source_pdf": pdf_name,
            "image_path": str(page.image_path),
        }
        nodes.append(TextNode(text=page.text, metadata=metadata))
    return nodes


class ColpaliRetriever(BaseRetriever):
    def __init__(
        self,
        nodes: List[TextNode],
        embeddings: List[torch.Tensor],
        model: ColQwen2_5,
        processor: ColQwen2_5_Processor,
        device: torch.device,
        dtype: torch.dtype,
        top_k: int = 3,
    ) -> None:
        super().__init__()
        self._nodes = nodes
        self._embeddings = embeddings
        self._model = model
        self._processor = processor
        self._device = device
        self._dtype = dtype
        self._top_k = top_k

    def _embed_query(self, query: str) -> torch.Tensor:
        batch_queries = self._processor.process_queries([query])
        batch_queries = move_to_device(batch_queries, self._device, self._dtype)
        with torch.no_grad():
            query_embedding = self._model(**batch_queries)
        mask = batch_queries["attention_mask"][0].bool()
        return query_embedding[0][mask].detach().cpu()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_tensor = self._embed_query(query_bundle.query_str)
        scores = self._processor.score_multi_vector([query_tensor], self._embeddings, device=self._device)[0]
        top_k = min(self._top_k, scores.numel())
        ranked_indices = torch.topk(scores, top_k).indices.tolist()
        return [
            NodeWithScore(node=self._nodes[idx], score=float(scores[idx]))
            for idx in ranked_indices
        ]


def extract_lama70b_gsm8k_metrics(text: str):
    pattern = r"Llama2-70B[^\n]*\n([0-9.]+)\s*\u00B1\s*([0-9.e-]+)\n([0-9.]+)\s*\u00B1\s*([0-9.e-]+)"
    match = re.search(pattern, text)
    if not match:
        return None
    single, single_err, ours, ours_err = match.groups()
    return {
        "single_accuracy": float(single),
        "single_std": float(single_err),
        "ensemble_accuracy": float(ours),
        "ensemble_std": float(ours_err),
    }


def collect_relevant_lines(text: str, keywords: Iterable[str], max_lines: int = 4) -> List[str]:
    lowered = [k.lower() for k in keywords]
    collected: List[str] = []
    for line in text.splitlines():
        if any(keyword in line.lower() for keyword in lowered):
            collected.append(line.strip())
        if len(collected) >= max_lines:
            break
    return collected


def main():
    parser = argparse.ArgumentParser(description="ColPali + LlamaIndex retrieval demo")
    parser.add_argument("--pdf-url", default=DEFAULT_PDF_URL, help="PDF to index")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="ColPali checkpoint")
    parser.add_argument("--pdf-path", default="data/2402.05120.pdf", help="Local PDF path")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for rendered pages")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Question to ask")
    parser.add_argument("--device", default=None, help="Force device (cpu, cuda, mps)")
    parser.add_argument("--top-k", type=int, default=3, help="Number of pages to display")
    parser.add_argument("--refresh-images", action="store_true", help="Re-render PDF pages")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    download_pdf(args.pdf_url, pdf_path)

    artifacts_dir = Path(args.artifacts_dir)
    image_dir = artifacts_dir / "images"

    pages = render_pdf_to_images(pdf_path, image_dir, refresh=args.refresh_images)
    nodes = build_nodes(pages, pdf_path)

    device = resolve_device(args.device)
    model, processor, dtype = load_colqwen(args.model_id, device)

    logging.info("Encoding %d pages", len(pages))
    page_embeddings = compute_page_embeddings(pages, model, processor, device, dtype)

    retriever = ColpaliRetriever(
        nodes=nodes,
        embeddings=page_embeddings,
        model=model,
        processor=processor,
        device=device,
        dtype=dtype,
        top_k=args.top_k,
    )

    logging.info("Running retrieval for question: %s", args.question)
    results = retriever.retrieve(args.question)

    if not results:
        logging.warning("No results found")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result.node.metadata
        page_number = metadata.get("page_number")
        snippet_lines = collect_relevant_lines(result.node.get_content(), ["Llama2-70B", "GSM8K", "accuracy"])
        print(f"Rank {rank}: page {page_number} (score={result.score:.4f})")
        for line in snippet_lines:
            print(f"    {line}")

    extracted = None
    source_page = None
    for result in results:
        parsed = extract_lama70b_gsm8k_metrics(result.node.get_content())
        if parsed:
            extracted = parsed
            source_page = result.node.metadata.get("page_number")
            break

    if extracted is None:
        for node in nodes:
            parsed = extract_lama70b_gsm8k_metrics(node.get_content())
            if parsed:
                extracted = parsed
                source_page = node.metadata.get("page_number")
                break

    if extracted:
        print()
        print("Llama2-70B GSM8K accuracy (from Table 2):")
        print(
            f" - Ensemble size 1 (single query): {extracted['single_accuracy'] * 100:.2f}% \u00B1 {extracted['single_std'] * 100:.2f}%"
        )
        print(
            f" - Ensemble size 40 (Agent Forest ensemble): {extracted['ensemble_accuracy'] * 100:.2f}% \u00B1 {extracted['ensemble_std'] * 100:.2f}%"
        )
        print(f"   Source: page {source_page} of {pdf_path.name}")
    else:
        logging.warning("Could not parse explicit accuracy values from retrieved pages")


if __name__ == "__main__":
    sys.exit(main())

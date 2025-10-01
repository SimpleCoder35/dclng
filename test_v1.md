"""Utility for converting textual documents through a Docling pipeline.

This module builds a multi-format Docling DocumentConverter that
is tuned for textual PDF/DOCX/PPTX inputs. The converter automatically
selects GPU acceleration when CUDA 12.1+ is present, and otherwise falls
back to CPU execution. The CLI can be used to batch-convert files into
plain text or Markdown while keeping the Docling ConversionResult objects
available for further processing.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence, Tuple

import torch
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.backend.mspowerpoint_backend import MsPowerpointDocumentBackend
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import (
    DOCLING_LAYOUT_EGRET_LARGE,
    DOCLING_LAYOUT_HERON,
)
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    EasyOcrOptions,
    LayoutOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, FormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline


_LOG = logging.getLogger(__name__)


def _detect_device(preferred: str = "cuda") -> str:
    """Return the best available accelerator for Docling models."""
    if preferred.startswith("cuda") and torch.cuda.is_available():
        dev_index = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(dev_index)
        capability = torch.cuda.get_device_capability(dev_index)
        _LOG.info(
            "Using CUDA device %s (compute capability %d.%d)",
            device_name,
            capability[0],
            capability[1],
        )
        return f"cuda:{dev_index}"

    if preferred in {"cpu", "auto"}:
        return "cpu"

    _LOG.warning("Falling back to CPU execution; CUDA is not available.")
    return "cpu"


def _make_accelerator_options(device: str) -> AcceleratorOptions:
    """Build accelerator options with a sensible thread count."""
    avail_cores = os.cpu_count() or 8
    # Keep at least a couple of cores free for other work.
    num_threads = max(2, avail_cores - 2)

    if device.startswith("cuda"):
        accelerator_device = AcceleratorDevice.CUDA
    else:
        accelerator_device = AcceleratorDevice.CPU

    return AcceleratorOptions(
        device=accelerator_device.value,
        num_threads=num_threads,
    )


def _select_layout_model(device: str):
    """Select a layout model tuned for the current accelerator."""
    if device.startswith("cuda"):
        return DOCLING_LAYOUT_EGRET_LARGE
    return DOCLING_LAYOUT_HERON


def build_converter(preferred_device: str = "cuda") -> Tuple[DocumentConverter, str]:
    """Create a DocumentConverter tuned for textual documents."""
    device = _detect_device(preferred_device)
    accelerator_options = _make_accelerator_options(device)

    layout_model = _select_layout_model(device)
    pdf_options = PdfPipelineOptions(
        accelerator_options=accelerator_options,
        layout_options=LayoutOptions(model_spec=layout_model),
        ocr_options=EasyOcrOptions(use_gpu=device.startswith("cuda")),
        do_ocr=True,
        do_table_structure=True,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        generate_page_images=False,
        generate_picture_images=False,
        generate_parsed_pages=False,
    )

    doc_like_options = ConvertPipelineOptions(
        accelerator_options=accelerator_options,
    )

    format_options = {
        InputFormat.PDF: FormatOption(
            pipeline_cls=StandardPdfPipeline,
            backend=DoclingParseV4DocumentBackend,
            pipeline_options=pdf_options,
        ),
        InputFormat.DOCX: FormatOption(
            pipeline_cls=SimplePipeline,
            backend=MsWordDocumentBackend,
            pipeline_options=doc_like_options,
        ),
        InputFormat.PPTX: FormatOption(
            pipeline_cls=SimplePipeline,
            backend=MsPowerpointDocumentBackend,
            pipeline_options=doc_like_options,
        ),
    }

    converter = DocumentConverter(
        allowed_formats=list(format_options.keys()),
        format_options=format_options,
    )

    return converter, device


def _normalize_whitespace(text: str) -> str:
    """Collapse horizontal whitespace and limit blank lines."""
    collapsed = re.sub(r"[\t ]+", " ", text)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed.strip()


def convert_documents(
    converter: DocumentConverter,
    sources: Iterable[Path],
    output_dir: Path | None = None,
    *,
    normalize_whitespace: bool = True,
    output_format: Literal["text", "markdown"] = "text",
) -> Iterator[Tuple[Path, str]]:
    """Convert documents and optionally persist text output."""
    output_format = output_format.lower()  # type: ignore[assignment]
    if output_format not in {"text", "markdown"}:
        raise ValueError(f"Unsupported output format: {output_format}")

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    for src in sources:
        res = converter.convert(src)
        if res.document is None:
            _LOG.warning("No document produced for %s", src)
            continue

        if output_format == "markdown":
            content = res.document.export_to_markdown()
        else:
            content = res.document.export_to_text()
            if normalize_whitespace:
                content = _normalize_whitespace(content)

        if output_dir:
            suffix = ".md" if output_format == "markdown" else ".txt"
            out_path = output_dir / (src.stem + suffix)
            out_path.write_text(content, encoding="utf-8")
            _LOG.info("Wrote %s", out_path)

        yield src, content


def _iter_input_paths(inputs: Sequence[Path], recurse: bool) -> Iterator[Path]:
    """Yield documents from the provided paths."""
    extensions = {".pdf", ".docx", ".pptx"}
    for entry in inputs:
        if entry.is_file() and entry.suffix.lower() in extensions:
            yield entry
        elif entry.is_dir():
            if recurse:
                for candidate in entry.rglob("*"):
                    if candidate.is_file() and candidate.suffix.lower() in extensions:
                        yield candidate
            else:
                for candidate in entry.iterdir():
                    if candidate.is_file() and candidate.suffix.lower() in extensions:
                        yield candidate
        else:
            _LOG.debug("Skipping non-document path %s", entry)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories containing PDF/DOCX/PPTX documents.",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        type=Path,
        default=None,
        help="Optional directory where converted outputs will be written.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even if CUDA is available.",
    )
    parser.add_argument(
        "--recurse",
        action="store_true",
        help="Recurse into directories when searching for documents.",
    )
    parser.add_argument(
        "--raw-text",
        action="store_true",
        help="Skip whitespace normalization when emitting plain text.",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=["text", "markdown"],
        help="Output representation to emit when writing files (default: text).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    converter, device = build_converter("cpu" if args.cpu else "cuda")
    _LOG.info("Docling converter ready (device=%s)", device)

    inputs = list(_iter_input_paths(args.paths, recurse=args.recurse))
    if not inputs:
        _LOG.warning("No supported documents found under %s", args.paths)
        return

    for _path, _text in convert_documents(
        converter=converter,
        sources=inputs,
        output_dir=args.output_dir,
        normalize_whitespace=not args.raw_text,
        output_format=args.format,
    ):
        _LOG.info("Processed %s", _path)


if __name__ == "__main__":
    main()

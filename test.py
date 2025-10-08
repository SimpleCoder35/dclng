# 📘 Docling Parallel Benchmark (GPU/CPU auto-tuning)

# This notebook will try multiple **workers** (processes) and **OMP** (inner CPU threads)
# and optionally toggle **page images**, then summarize the throughput & stability so
# you can pick the fastest stable combo (great for NVIDIA L4).

# **Start** with 5–10 PDFs. On an L4 24GB, `workers=2`, `OMP=1` are strong defaults.

# --- Optional: installs (uncomment if needed) ---
# %pip install -U docling docling-core pypdfium2 rapidocr-onnxruntime
# # PyTorch CUDA wheel (adjust CUDA version if needed). See: https://pytorch.org/get-started/locally/
# # Example for CUDA 12.1 wheels:
# # %pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# %pip install -U pandas matplotlib tqdm ipywidgets
# # If widgets don't render, also run (in a terminal once):
# # jupyter nbextension enable --py widgetsnbextension

## Cell 3 — Imports & pathsimport os
import csv
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 200)

PDF_DIR = "pdfs"       # ← change to your PDF folder
OUT_DIR = "out_bench"  # results + markdown outputs

# Detect CUDA / GPU (optional)
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if HAS_CUDA else "CPU only"
except Exception:
    HAS_CUDA = False
    GPU_NAME = "CPU only"

print(f"CUDA available: {HAS_CUDA}  |  Device: {GPU_NAME}")

## Cell 4 — Write helper library (robust for Windows / ProcessPool in notebooks)
benchlib_code = r'''# docling_benchlib.py
import os, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import ImageRefMode

CONVERTER = None

def init_env(omp_threads: int):
    # Keep inner thread fan-out tame so outer process parallelism scales
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    try:
        import torch
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            try:
                import torch.backends.cuda as cuda_b
                cuda_b.matmul.allow_tf32 = True
            except Exception:
                pass
    except Exception:
        pass

def get_converter(generate_images: bool):
    global CONVERTER
    if CONVERTER is None:
        pdf_opts = PdfPipelineOptions()
        pdf_opts.generate_page_images = generate_images
        CONVERTER = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=DoclingParseV4DocumentBackend,
                pipeline_options=pdf_opts
            )
        })
    return CONVERTER

def convert_one(pdf_path: str, out_dir: str, generate_images: bool, omp_threads: int):
    """
    Returns a dict with metrics for this file conversion.
    """
    t0 = time.time()
    init_env(omp_threads)
    conv = get_converter(generate_images=generate_images)

    gpu_alloc_mb = 0
    gpu_reserved_mb = 0
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    err = None
    out_md = None
    try:
        res = conv.convert(pdf_path)
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        out_md = str(out_dir_p / (Path(pdf_path).stem + ".md"))
        res.document.save_as_markdown(out_md, image_mode=ImageRefMode.PLACEHOLDER)

        if has_cuda:
            import torch
            try:
                gpu_alloc_mb = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
                gpu_reserved_mb = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
            except Exception:
                pass
    except Exception as e:
        err = str(e)

    dt = time.time() - t0
    return {
        "src": pdf_path,
        "out": out_md,
        "sec": round(dt, 3),
        "error": err,
        "gpu_alloc_mb": gpu_alloc_mb,
        "gpu_reserved_mb": gpu_reserved_mb,
    }

def run_combo(files, combo_out_dir, workers, images, omp_threads):
    t0 = time.time()
    results = []
    succeeded = 0
    failed = 0
    max_gpu_alloc = 0
    max_gpu_reserved = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(convert_one, f, combo_out_dir, images, omp_threads): f
            for f in files
        }
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if r["error"]:
                failed += 1
            else:
                succeeded += 1
                if r["gpu_alloc_mb"] > max_gpu_alloc:
                    max_gpu_alloc = r["gpu_alloc_mb"]
                if r["gpu_reserved_mb"] > max_gpu_reserved:
                    max_gpu_reserved = r["gpu_reserved_mb"]

    total_s = time.time() - t0
    files_done = succeeded + failed
    throughput_fpm = (files_done / total_s) * 60 if total_s > 0 else 0.0
    tpf = total_s / files_done if files_done else 0.0

    first_sec = None
    if results:
        ok_secs = [r["sec"] for r in results if not r["error"]]
        if ok_secs:
            first_sec = min(ok_secs)

    avg_nonfirst = None
    if results and first_sec is not None:
        nonfirst = [r["sec"] for r in results if not r["error"] and r["sec"] != first_sec]
        if nonfirst:
            avg_nonfirst = sum(nonfirst) / len(nonfirst)

    summary = {
        "workers": workers,
        "images": int(images),
        "omp_threads": omp_threads,
        "files": files_done,
        "ok": succeeded,
        "fail": failed,
        "total_sec": round(total_s, 2),
        "sec_per_file": round(tpf, 2),
        "files_per_min": round(throughput_fpm, 2),
        "first_file_sec": round(first_sec, 2) if first_sec is not None else None,
        "avg_nonfirst_sec": round(avg_nonfirst, 2) if avg_nonfirst is not None else None,
        "max_gpu_alloc_mb": max_gpu_alloc,
        "max_gpu_reserved_mb": max_gpu_reserved,
    }
    return summary, results
'''
with open("docling_benchlib.py", "w", encoding="utf-8") as f:
    f.write(benchlib_code)

print("Wrote docling_benchlib.py")

## cell 5

from pathlib import Path
from docling_benchlib import run_combo

# --- Parameters ---
WORKERS_LIST = [1, 2, 3]   # try 1–3 workers; increase only if VRAM/CPU allow
OMP_LIST     = [1]         # inner CPU threads per worker
TEST_IMAGES  = False       # True if you need page images in the pipeline
MAX_FILES    = None        # e.g., 10 to limit sample size

# Gather files
files = sorted(str(p) for p in Path(PDF_DIR).glob("*.pdf"))
if MAX_FILES:
    files = files[:MAX_FILES]

if not files:
    raise SystemExit(f"No PDFs found in {PDF_DIR!r}")

print(f"Found {len(files)} PDF(s). Running grid...")

### Cell 6 — Run grid, save CSVs, show summary

summary_rows = []
perfile_rows = []

images_list = [False, True] if TEST_IMAGES else [False]
out_root = Path(OUT_DIR)
out_root.mkdir(parents=True, exist_ok=True)

for images in images_list:
    for omp_threads in OMP_LIST:
        for workers in WORKERS_LIST:
            combo_tag = f"w{workers}_omp{omp_threads}_img{int(images)}"
            combo_out = str(out_root / combo_tag)

            summary, per = run_combo(
                files=files,
                combo_out_dir=combo_out,
                workers=workers,
                images=images,
                omp_threads=omp_threads,
            )

            summary_rows.append(summary)
            for r in per:
                r["workers"] = workers
                r["images"] = int(images)
                r["omp_threads"] = omp_threads
                perfile_rows.append(r)

summary_df = pd.DataFrame(summary_rows).sort_values(
    ["files_per_min", "sec_per_file"], ascending=[False, True], ignore_index=True
)
perfile_df = pd.DataFrame(perfile_rows)

# Save CSVs
summary_csv = out_root / "benchmark_summary.csv"
perfile_csv = out_root / "per_file_metrics.csv"
summary_df.to_csv(summary_csv, index=False)
perfile_df.to_csv(perfile_csv, index=False)

print("== Completed grid ==")
print("Summary CSV :", summary_csv)
print("Per-file CSV:", perfile_csv)

summary_df

## Cell 7 — Plot (matplotlib, single plot)

plt.figure()
for images in sorted(summary_df["images"].unique()):
    for omp in sorted(summary_df["omp_threads"].unique()):
        subset = summary_df[(summary_df["images"] == images) & (summary_df["omp_threads"] == omp)]
        subset = subset.sort_values("workers")
        if not subset.empty:
            label = f"img={images}, omp={omp}"
            plt.plot(subset["workers"], subset["files_per_min"], marker="o", label=label)

plt.xlabel("workers")
plt.ylabel("files_per_min")
plt.title("Docling throughput by workers (higher is better)")
plt.legend()
plt.grid(True)
plt.show()


## Cell 8 — Simple “best config” recommender

def recommend_best(summary_df: pd.DataFrame, total_files: int, vram_budget_mb: int = 23000):
    """
    Pick the row with highest files_per_min that:
      - processed all files (ok == total_files)
      - stayed under a VRAM budget (max_gpu_reserved_mb < vram_budget_mb), if present
    Fallback to overall best files_per_min if no row matches constraints.
    """
    cand = summary_df.copy()
    cand = cand[cand["ok"] == total_files]
    if "max_gpu_reserved_mb" in cand.columns:
        cand = cand[cand["max_gpu_reserved_mb"] < vram_budget_mb]
    if cand.empty:
        cand = summary_df.copy()
    best = cand.sort_values(["files_per_min", "sec_per_file"], ascending=[False, True]).head(1)
    return best

best_row = recommend_best(summary_df, total_files=len(files), vram_budget_mb=23000)
print("Recommended configuration:")
best_row

## (Optional) Cell 9 — Run the chosen config for a full output folder

# Re-run with the recommended settings to produce a single "final_out" directory of markdowns
final_out = Path(OUT_DIR) / "final_out"
final_out.mkdir(parents=True, exist_ok=True)

w = int(best_row["workers"].iloc[0])
omp = int(best_row["omp_threads"].iloc[0])
img = bool(int(best_row["images"].iloc[0]))

print(f"Running final conversion with workers={w}, omp={omp}, images={img} → {final_out}")
from docling_benchlib import run_combo
summary, _ = run_combo(files, str(final_out), w, img, omp)
summary


# How to use

# Put 5–10 representative PDFs in pdfs/ (or change PDF_DIR).

# Run cells 2 → 8.

# Check summary_df (and chart). The recommender will propose a safe, fast combo for your L4.

# (Optional) Run Cell 9 to produce a unified output folder with those settings.

# If you want, I can also add an ipywidgets UI (dropdowns + “Run”) on top of this, but the cells above will get you optimal settings quickly.

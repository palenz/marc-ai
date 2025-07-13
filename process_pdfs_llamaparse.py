#!/usr/bin/env python
"""process_pdfs_llamaparse.py

Bulk-convert every PDF in data/pdfs/ → plain-text files in data/text/ via LlamaParse.

Requirements
------------
1. pip install llama-parse tqdm
2. Set an environment variable LLAMA_CLOUD_API_KEY (or replace the placeholder below).

Usage
-----
$ export LLAMA_CLOUD_API_KEY="<your-key>"
$ python process_pdfs_llamaparse.py

The script will create `data/text/` if it does not yet exist and will skip conversion
if the corresponding .txt already exists (idempotent, safe to rerun).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

from llama_parse import LlamaParse
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration – edit as needed
# ---------------------------------------------------------------------------
PDF_DIR = Path("data/pdfs")
OUT_DIR = Path("data/text")
API_KEY = "llx"  # Or replace with your key string
RESULT_TYPE = "text"  # “markdown” is also supported
LANGUAGE = "es"  # Spanish language for parsing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_pdfs(folder: Path) -> Iterable[Path]:
    """Yield all .pdf files in *folder* (non-recursive)."""
    if not folder.exists():
        raise FileNotFoundError(f"Input directory not found: {folder.resolve()}")
    yield from sorted(p for p in folder.iterdir() if p.suffix.lower() == ".pdf")


def save_text(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_single(pdf_path: Path, parser: LlamaParse) -> None:
    """Parse *pdf_path* with *parser* and write its text next to OUT_DIR."""
    target_txt = OUT_DIR / (pdf_path.stem + ".txt")
    if target_txt.exists():
        # Skip already processed files for efficiency / restart safety.
        return

    try:
        docs = parser.load_data(str(pdf_path))
    except Exception as exc:
        print(f"❌ Failed to parse {pdf_path.name}: {exc}", file=sys.stderr)
        return

    # LlamaParse returns a list of Document objects. Concatenate their text.
    joined_text = "\n\n".join(getattr(doc, "text", str(doc)) for doc in docs if doc)
    save_text(target_txt, joined_text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        sys.exit("Environment variable LLAMA_CLOUD_API_KEY not found – please set it or edit the script to hard-code your key.")

    parser = LlamaParse(api_key=API_KEY, result_type=RESULT_TYPE, language=LANGUAGE)

    pdf_files = list(iter_pdfs(PDF_DIR))
    if not pdf_files:
        sys.exit(f"No PDF files found in {PDF_DIR.resolve()}")

    print(f"📑 Found {len(pdf_files)} PDF(s). Parsing with LlamaParse…")
    for pdf_path in tqdm(pdf_files, unit="pdf"):
        process_single(pdf_path, parser)

    print("✅ All done! Converted files are in", OUT_DIR.resolve())


if __name__ == "__main__":
    main() 
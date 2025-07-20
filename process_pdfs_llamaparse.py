#!/usr/bin/env python
"""process_pdfs_unstructured.py

Bulk-convert every PDF in /Users/juan/Desktop/pdfs/ → chunks with embeddings using unstructured.
Saves vector embeddings to embeddings.pkl for later use in Streamlit app.

Requirements
------------
1. pip install unstructured openai numpy tqdm
2. Set OPENAI_API_KEY environment variable
3. Install tesseract with Spanish language support

Usage
-----
$ export OPENAI_API_KEY="<your-key>"
$ python process_pdfs_llamaparse.py

The script will create embeddings.pkl with all chunks and their vector embeddings.
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from unstructured.partition.pdf import partition_pdf
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PDF_DIR = Path("/Users/juan/Desktop/pdfs")
EMBEDDINGS_FILE = "embeddings.pkl"
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

# Unstructured settings
UNSTRUCTURED_STRATEGY = "fast"  # Avoid onnxruntime issues
CHUNK_MAX_CHARS = 10000
CHUNK_COMBINE_CHARS = 2000
CHUNK_NEW_AFTER_CHARS = 6000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_pdfs(folder: Path) -> List[Path]:
    """Return all .pdf files in *folder*."""
    if not folder.exists():
        raise FileNotFoundError(f"Input directory not found: {folder.resolve()}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() == ".pdf")


def process_single_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Process a single PDF and return chunks with metadata."""
    try:
        chunks = partition_pdf(
            filename=str(pdf_path),
            strategy=UNSTRUCTURED_STRATEGY,
            ocr_languages=["spa"],  # Enable Spanish OCR for better text extraction
            
            chunking_strategy="by_title",
            max_characters=CHUNK_MAX_CHARS,
            combine_text_under_n_chars=CHUNK_COMBINE_CHARS,
            new_after_n_chars=CHUNK_NEW_AFTER_CHARS,
        )
        
        # Extract document title from the first title element
        document_title = None
        for chunk in chunks[:5]:  # Check first 5 chunks for title
            if hasattr(chunk, 'category') and chunk.category == 'Title':
                document_title = str(chunk).strip()
                if len(document_title) > 10 and len(document_title) < 200:  # Reasonable title length
                    break
        
        # Fallback to filename if no title found
        if not document_title:
            document_title = pdf_path.stem.replace('_', ' ').replace('-', ' ')
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            # Extract metadata
            metadata = chunk.metadata.to_dict() if hasattr(chunk.metadata, 'to_dict') else {}
            
            # Get page number (if available)
            page_number = metadata.get('page_number', 1)
            
            # Get category/type information
            element_category = getattr(chunk, 'category', 'Unknown')
            
            # Create chunk data
            chunk_data = {
                'text': str(chunk),
                'source_file': pdf_path.name,
                'source_path': str(pdf_path),
                'document_title': document_title,
                'chunk_id': i,
                'page_number': page_number,
                'element_type': str(type(chunk).__name__),
                'element_category': element_category,
                'metadata': metadata
            }
            
            # Only include chunks with meaningful text
            if len(chunk_data['text'].strip()) > 50:
                processed_chunks.append(chunk_data)
                
        return processed_chunks
        
    except Exception as exc:
        print(f"❌ Failed to process {pdf_path.name}: {exc}", file=sys.stderr)
        return []


def create_embeddings(chunks: List[Dict[str, Any]], client: OpenAI) -> np.ndarray:
    """Create embeddings for all chunks."""
    texts = [chunk['text'] for chunk in chunks]
    embeddings = []
    
    print(f"Creating embeddings for {len(texts)} chunks...")
    
    # Process in batches
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
        batch_texts = texts[i:i + BATCH_SIZE]
        
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch_texts
            )
            batch_embeddings = [d.embedding for d in response.data]
            embeddings.extend(batch_embeddings)
            
        except Exception as exc:
            print(f"❌ Failed to create embeddings for batch {i//BATCH_SIZE + 1}: {exc}")
            # Add zero embeddings as fallback
            embeddings.extend([[0.0] * 1536] * len(batch_texts))
    
    # Convert to numpy array and normalize for cosine similarity
    embeddings_array = np.array(embeddings, dtype="float32")
    embeddings_array /= np.linalg.norm(embeddings_array, axis=1, keepdims=True) + 1e-10
    
    return embeddings_array


def save_embeddings_data(chunks: List[Dict[str, Any]], embeddings: np.ndarray, output_file: str):
    """Save chunks and embeddings to pickle file."""
    data = {
        'chunks': chunks,
        'embeddings': embeddings,
        'embed_model': EMBED_MODEL,
        'total_chunks': len(chunks),
        'embedding_dim': embeddings.shape[1] if len(embeddings) > 0 else 0
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Saved {len(chunks)} chunks and embeddings to {output_file}")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def main() -> None:
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        sys.exit("❌ Environment variable OPENAI_API_KEY not found – please set it.")
    
    client = OpenAI(api_key=openai_api_key)
    
    # Find PDF files
    pdf_files = iter_pdfs(PDF_DIR)
    if not pdf_files:
        sys.exit(f"❌ No PDF files found in {PDF_DIR.resolve()}")
    
    print(f"📑 Found {len(pdf_files)} PDF(s) in {PDF_DIR}")
    print("🔄 Processing PDFs with unstructured...")
    
    # Process all PDFs
    all_chunks = []
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        chunks = process_single_pdf(pdf_path)
        all_chunks.extend(chunks)
        print(f"  📄 {pdf_path.name}: {len(chunks)} chunks")
    
    if not all_chunks:
        sys.exit("❌ No chunks extracted from PDFs")
    
    print(f"📊 Total chunks extracted: {len(all_chunks)}")
    
    # Create embeddings
    embeddings = create_embeddings(all_chunks, client)
    
    # Save to file
    save_embeddings_data(all_chunks, embeddings, EMBEDDINGS_FILE)
    
    print("✅ All done! Vector embeddings saved to embeddings.pkl")
    print(f"📈 Summary:")
    print(f"  - PDFs processed: {len(pdf_files)}")
    print(f"  - Total chunks: {len(all_chunks)}")
    print(f"  - Embedding dimensions: {embeddings.shape[1] if len(embeddings) > 0 else 0}")


if __name__ == "__main__":
    main() 
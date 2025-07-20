# Marc AI - RAG System with Unstructured

A Retrieval-Augmented Generation (RAG) system for answering questions about Spanish technical regulations using unstructured document processing.

## Features

- **Better PDF Processing**: Uses `unstructured` library for superior text extraction with page numbers and document metadata
- **Pre-computed Embeddings**: Processes PDFs once and saves embeddings to avoid recomputation
- **Spanish Language Support**: Optimized for Spanish technical documents
- **Rich Metadata**: Preserves document structure, page numbers, and element types
- **Interactive Chat Interface**: Streamlit-based chat with source attribution
- **Debug Information**: Shows similarity scores and source documents for transparency

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Prepare Your PDFs

Place your PDF files in `/Users/juan/Desktop/pdfs/` (or update the path in `process_pdfs_llamaparse.py`)

### 4. Test Setup

```bash
python setup_and_run.py
```

## Usage

### Step 1: Process PDFs and Create Embeddings

```bash
python process_pdfs_llamaparse.py
```

This will:
- Process all PDFs in your directory using unstructured
- Extract chunks with proper metadata (page numbers, document info, element types)
- Create vector embeddings using OpenAI's `text-embedding-3-small`
- Save everything to `embeddings.pkl`

### Step 2: Run the Chat Interface

```bash
streamlit run streamlit_app.py
```

This will start the Streamlit app where you can:
- Ask questions about your technical documents
- Get answers with proper source attribution
- See debug information about which documents were consulted
- View statistics about your document collection

## How It Works

### Document Processing (process_pdfs_llamaparse.py)

1. **PDF Parsing**: Uses `unstructured.partition.pdf` with:
   - `strategy="fast"` to avoid onnxruntime issues
   - `chunking_strategy="by_title"` for intelligent chunking
   - Configurable chunk sizes (10k max chars, combine under 2k, new after 6k)

2. **Metadata Extraction**: Captures:
   - Source file name and path
   - Page numbers
   - Element types (Title, Text, Table, etc.)
   - Original unstructured metadata

3. **Embedding Creation**: 
   - Processes chunks in batches of 100
   - Uses OpenAI's `text-embedding-3-small`
   - L2-normalizes for cosine similarity
   - Saves to pickle file for reuse

### Chat Interface (streamlit_app.py)

1. **Loading**: Loads pre-computed embeddings from `embeddings.pkl`
2. **Search**: For each query:
   - Creates embedding for the user question
   - Finds top 5 most similar chunks using cosine similarity
   - Formats chunks with source attribution
3. **Generation**: Uses GPT-4o-mini with context and system prompt
4. **Display**: Shows answer with debug information about sources

## Configuration

### Unstructured Settings (in process_pdfs_llamaparse.py)

```python
UNSTRUCTURED_STRATEGY = "fast"  # Avoid onnxruntime issues
CHUNK_MAX_CHARS = 10000
CHUNK_COMBINE_CHARS = 2000
CHUNK_NEW_AFTER_CHARS = 6000
```

### Embedding Settings

```python
EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
```

## Troubleshooting

### Common Issues

1. **Missing Spanish Language Support**:
   ```bash
   brew install tesseract-lang
   export TESSDATA_PREFIX=/opt/homebrew/share/
   ```

2. **PDF Processing Errors**: Check that PDFs are not corrupted or password-protected

3. **Memory Issues**: Reduce `BATCH_SIZE` if you encounter memory problems

4. **Missing Embeddings**: Run `process_pdfs_llamaparse.py` first before starting the Streamlit app

### Debug Information

The Streamlit app provides debug information including:
- Similarity scores for retrieved chunks
- Source document names and page numbers
- Element types from unstructured
- Full text of retrieved chunks

## File Structure

```
├── process_pdfs_llamaparse.py  # PDF processing and embedding creation
├── streamlit_app.py           # Chat interface
├── requirements.txt           # Dependencies
├── setup_and_run.py          # Setup checker
├── README.md                 # This file
├── embeddings.pkl            # Generated embeddings file (after processing)
└── new.py                    # Original unstructured experiments
```

## Advantages Over Previous Version

1. **Better Text Extraction**: Unstructured handles complex PDFs better than simple text conversion
2. **Preserved Metadata**: Page numbers and document structure are maintained
3. **No Reprocessing**: Embeddings are created once and reused
4. **Spanish Optimization**: Better handling of Spanish text and technical documents
5. **Transparency**: Debug information shows exactly which sources were used
6. **Scalability**: Can handle large document collections efficiently

## Future Improvements

- [ ] Support for other document formats (Word, HTML)
- [ ] Advanced filtering by document type or date
- [ ] Semantic chunking based on document structure
- [ ] Integration with local embedding models
- [ ] Support for images and tables in responses

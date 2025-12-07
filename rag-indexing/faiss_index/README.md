# FAISS to TensorFlow Projector Converter

Convert FAISS vector index and pickle metadata to TSV format compatible with [TensorFlow's Embedding Projector](https://projector.tensorflow.org/).

## Overview

This tool extracts embeddings and metadata from FAISS vector databases created by LangChain and exports them in a format that can be visualized using TensorFlow's Embedding Projector.

## Files

- `convert.py` - Main conversion script
- `pyproject.toml` - Project dependencies managed by UV
- `index.faiss` - FAISS vector index (input)
- `index.pkl` - Pickle metadata file (input)
- `vectors.tsv` - Exported embeddings (output)
- `metadata.tsv` - Exported document metadata (output)

## Requirements

- Python 3.10+
- UV package manager

## Installation

Install dependencies using UV:

```bash
uv sync
```

This will install:
- `faiss-cpu` - FAISS library for vector search
- `numpy` - Numerical operations
- `langchain-core` - LangChain core components
- `langchain-community` - LangChain community integrations

## Usage

### Basic Usage

Run the conversion script:

```bash
uv run python convert.py
```

This will:
1. Load `index.faiss` (FAISS vector index)
2. Load `index.pkl` (metadata from LangChain FAISS vectorstore)
3. Extract 23 vectors of dimension 1024
4. Export to `vectors.tsv` (~287 KB)
5. Export metadata to `metadata.tsv` (~13 KB)

### Output Files

#### vectors.tsv
- Tab-separated values
- Each row is one embedding vector
- Dimensions: 23 rows × 1024 columns
- No header row

#### metadata.tsv
- Tab-separated values with header
- Columns:
  - `index` - Vector index (0-22)
  - `doc_id` - Document UUID
  - `page` - PDF page number
  - `source` - Source filename (samsung-s22-camera.pdf)
  - `content_preview` - First 200 characters of document chunk
  - `content_length` - Full content length in characters

## Visualizing in TensorFlow Projector

1. Visit https://projector.tensorflow.org/
2. Click **"Load"** button in the left panel
3. Upload files:
   - **Load data**: Select `vectors.tsv`
   - **Load metadata**: Select `metadata.tsv`
4. Explore your embeddings!

### Visualization Features

Once loaded, you can:
- **3D/2D projection**: View embeddings using PCA, t-SNE, or UMAP
- **Search**: Find similar documents by content
- **Color by metadata**: Color points by page, source, etc.
- **Nearest neighbors**: Click a point to see similar documents
- **Custom projection**: Adjust visualization parameters

## Customization

### Different Input Files

Edit `convert.py` to specify different paths:

```python
convert_faiss_to_projector(
    faiss_path="path/to/your/index.faiss",
    pickle_path="path/to/your/index.pkl",
    output_dir="output"
)
```

### Additional Metadata Fields

To export more metadata, modify the `extract_metadata_list()` function in `convert.py`:

```python
metadata_list.append({
    'index': str(idx),
    'doc_id': str(doc_id),
    'page': str(doc_metadata.get('page', 'N/A')),
    'source': str(doc_metadata.get('source', 'N/A')),
    'content_preview': content_preview,
    'content_length': str(len(content)),
    # Add your custom fields:
    'custom_field': str(doc_metadata.get('custom_field', 'N/A'))
})
```

## Technical Details

### FAISS Index Structure

The FAISS index stores:
- **Vectors**: High-dimensional embeddings (1024D in this case)
- **Index type**: Flat index for exact similarity search
- **Total vectors**: 23 document chunks from Samsung S22 camera PDF

### Pickle Metadata Structure

LangChain's FAISS vectorstore saves metadata as a tuple:
```python
(docstore, index_to_docstore_id)
```

Where:
- `docstore`: InMemoryDocstore with LangChain Document objects
- `index_to_docstore_id`: Mapping from vector index to document ID

### Conversion Process

1. **Load FAISS index** - Uses `faiss.read_index()`
2. **Load pickle data** - Unpickles LangChain objects
3. **Reconstruct vectors** - Calls `index.reconstruct(i)` for each vector
4. **Extract documents** - Iterates through docstore
5. **Export to TSV** - Writes tab-separated files

## Example Output

```
$ uv run python convert.py

================================================================================
FAISS to TensorFlow Projector Converter
================================================================================
Loading FAISS index from: index.faiss
Index loaded - Total vectors: 23, Dimension: 1024
Loading metadata from: index.pkl
Metadata loaded
  Type: <class 'tuple'>
  Structure: (docstore, index_to_docstore_id)
  Documents: 23 entries

Extracting vectors from index...
Extracted 23 vectors of dimension 1024

Extracting metadata...
Extracted metadata for 23 documents

Saving vectors to: vectors.tsv
Saved 23 vectors
  File size: 0.28 MB

Saving metadata to: metadata.tsv
Saved metadata for 23 documents
  Columns: index, doc_id, page, source, content_preview, content_length
  File size: 12.19 KB

================================================================================
Conversion Complete!
================================================================================
```

## Troubleshooting

### ModuleNotFoundError

If you see missing module errors, ensure all dependencies are installed:
```bash
uv sync
```

### Memory Issues

For very large indices (millions of vectors), consider processing in batches:
- Modify `extract_vectors()` to process chunks
- Write TSV files incrementally

### Encoding Errors

The script uses UTF-8 encoding for TSV files. If you encounter encoding issues:
- Check that your terminal supports UTF-8
- Ensure input documents are properly encoded

## Related Projects

- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [LangChain](https://github.com/langchain-ai/langchain) - LLM application framework
- [TensorFlow Projector](https://projector.tensorflow.org/) - Embedding visualization

## License

This tool is provided as-is for educational purposes.

# -*- coding: utf-8 -*-
"""
FAISS Index to TensorFlow Projector Converter

This script converts FAISS vector index and pickle metadata to TSV format
compatible with TensorFlow's Embedding Projector (https://projector.tensorflow.org/)

Output files:
- vectors.tsv: Tab-separated embedding vectors
- metadata.tsv: Tab-separated metadata (document content, page, source)
"""

import pickle
import faiss
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_faiss_index(index_path: str = "index.faiss") -> faiss.Index:
    """
    Load FAISS index from file

    Args:
        index_path: Path to the FAISS index file

    Returns:
        Loaded FAISS index
    """
    print(f"Loading FAISS index from: {index_path}")
    index = faiss.read_index(index_path)
    print(f"Index loaded - Total vectors: {index.ntotal}, Dimension: {index.d}")
    return index


def load_pickle_metadata(pickle_path: str = "index.pkl") -> Dict:
    """
    Load metadata from pickle file

    Args:
        pickle_path: Path to the pickle metadata file

    Returns:
        Loaded metadata dictionary
    """
    print(f"Loading metadata from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    print(f"Metadata loaded")
    print(f"  Type: {type(data)}")

    # FAISS saves as tuple: (docstore, index_to_docstore_id)
    if isinstance(data, tuple) and len(data) == 2:
        docstore, index_to_docstore_id = data
        metadata = {
            'docstore': docstore,
            'index_to_docstore_id': index_to_docstore_id
        }
        print(f"  Structure: (docstore, index_to_docstore_id)")
        if hasattr(docstore, '_dict'):
            print(f"  Documents: {len(docstore._dict)} entries")
    elif isinstance(data, dict):
        metadata = data
        print(f"  Keys: {list(metadata.keys())}")
    else:
        print(f"  Warning: Unexpected pickle format")
        metadata = {'data': data}

    return metadata


def extract_vectors(index: faiss.Index) -> np.ndarray:
    """
    Extract all vectors from FAISS index

    Args:
        index: FAISS index

    Returns:
        Numpy array of vectors (shape: [n_vectors, dimension])
    """
    print(f"\nExtracting vectors from index...")
    n_total = index.ntotal
    dimension = index.d

    # Reconstruct all vectors from the index
    vectors = np.zeros((n_total, dimension), dtype=np.float32)

    for i in range(n_total):
        vectors[i] = index.reconstruct(int(i))

    print(f"Extracted {n_total} vectors of dimension {dimension}")
    return vectors


def extract_metadata_list(metadata: Dict) -> List[Dict[str, str]]:
    """
    Extract document metadata as a list of dictionaries

    Args:
        metadata: Metadata dictionary from pickle file

    Returns:
        List of metadata dictionaries for each vector
    """
    print(f"\nExtracting metadata...")
    metadata_list = []

    # Extract document store
    if 'docstore' in metadata:
        docstore = metadata['docstore']

        # Get index to docstore ID mapping
        index_to_docstore_id = metadata.get('index_to_docstore_id', {})

        # Extract documents
        if hasattr(docstore, '_dict'):
            docs = docstore._dict

            # Iterate through documents in order of index
            for idx in sorted(index_to_docstore_id.keys()):
                doc_id = index_to_docstore_id[idx]
                if doc_id in docs:
                    doc = docs[doc_id]

                    # Extract document content and metadata
                    content = getattr(doc, 'page_content', '')
                    doc_metadata = getattr(doc, 'metadata', {})

                    # Clean content for TSV (remove tabs and newlines)
                    content_clean = content.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                    content_preview = content_clean[:200] + '...' if len(content_clean) > 200 else content_clean

                    metadata_list.append({
                        'index': str(idx),
                        'doc_id': str(doc_id),
                        'page': str(doc_metadata.get('page', 'N/A')),
                        'source': str(doc_metadata.get('source', 'N/A')),
                        'content_preview': content_preview,
                        'content_length': str(len(content))
                    })

    print(f"Extracted metadata for {len(metadata_list)} documents")
    return metadata_list


def save_vectors_tsv(vectors: np.ndarray, output_path: str = "vectors.tsv") -> None:
    """
    Save vectors to TSV file for TensorFlow Projector

    Args:
        vectors: Numpy array of vectors
        output_path: Output file path
    """
    print(f"\nSaving vectors to: {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, vector in enumerate(vectors):
            # Write vector as tab-separated values
            vector_str = '\t'.join(map(str, vector))
            f.write(vector_str + '\n')

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(vectors)} vectors", end='\r')

    print(f"\nSaved {len(vectors)} vectors")

    # Print file size
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")


def save_metadata_tsv(metadata_list: List[Dict[str, str]], output_path: str = "metadata.tsv") -> None:
    """
    Save metadata to TSV file for TensorFlow Projector

    Args:
        metadata_list: List of metadata dictionaries
        output_path: Output file path
    """
    print(f"\nSaving metadata to: {output_path}")

    if not metadata_list:
        print("Warning: No metadata to save")
        return

    # Get headers from first entry
    headers = list(metadata_list[0].keys())

    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        f.write('\t'.join(headers) + '\n')

        # Write data rows
        for i, meta in enumerate(metadata_list):
            row = '\t'.join(meta.get(h, 'N/A') for h in headers)
            f.write(row + '\n')

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(metadata_list)} entries", end='\r')

    print(f"\nSaved metadata for {len(metadata_list)} documents")
    print(f"  Columns: {', '.join(headers)}")

    # Print file size
    file_size_kb = Path(output_path).stat().st_size / 1024
    print(f"  File size: {file_size_kb:.2f} KB")


def convert_faiss_to_projector(
    faiss_path: str = "index.faiss",
    pickle_path: str = "index.pkl",
    output_dir: str = "."
) -> None:
    """
    Main conversion function

    Args:
        faiss_path: Path to FAISS index file
        pickle_path: Path to pickle metadata file
        output_dir: Output directory for TSV files
    """
    print("=" * 80)
    print("FAISS to TensorFlow Projector Converter")
    print("=" * 80)

    # 1. Load FAISS index
    index = load_faiss_index(faiss_path)

    # 2. Load pickle metadata
    metadata = load_pickle_metadata(pickle_path)

    # 3. Extract vectors
    vectors = extract_vectors(index)

    # 4. Extract metadata
    metadata_list = extract_metadata_list(metadata)

    # 5. Save vectors to TSV
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True)

    vectors_path = output_dir_path / "vectors.tsv"
    save_vectors_tsv(vectors, str(vectors_path))

    # 6. Save metadata to TSV
    metadata_path = output_dir_path / "metadata.tsv"
    save_metadata_tsv(metadata_list, str(metadata_path))

    # 7. Print summary
    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  1. {vectors_path.absolute()}")
    print(f"  2. {metadata_path.absolute()}")
    print(f"\nNext steps:")
    print(f"  1. Visit https://projector.tensorflow.org/")
    print(f"  2. Click 'Load' button in the left panel")
    print(f"  3. Upload both TSV files:")
    print(f"     - vectors.tsv (as 'Data')")
    print(f"     - metadata.tsv (as 'Metadata')")
    print(f"  4. Explore your embeddings!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run conversion with default paths
    convert_faiss_to_projector()

"""
FAISS Semantic Search with Partitioning

This module implements semantic search using FAISS (Facebook AI Similarity Search)
with IVF (Inverted File Index) partitioning for optimized search performance.

The implementation uses:
- IndexIVFFlat: Combines flat L2 distance with Voronoi cell partitioning
- Sentence transformers for encoding text to embeddings
- Product quantization concepts for efficient similarity search

Author: Auto-generated
Date: 2026-01-03
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_embeddings(filepath="sentence_embeddings.npy"):
    """
    Load pre-computed sentence embeddings from a numpy file.

    Args:
        filepath (str): Path to the numpy file containing embeddings.
                       Default: "sentence_embeddings.npy"

    Returns:
        np.ndarray: Array of sentence embeddings with shape (n_sentences, embedding_dim)

    Raises:
        FileNotFoundError: If the embeddings file doesn't exist
    """
    print("Loading embeddings...")
    embeddings = np.load(filepath)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings


def load_sentences(filepath="sentences.txt"):
    """
    Load sentences from a text file.

    Args:
        filepath (str): Path to the text file containing sentences (one per line).
                       Default: "sentences.txt"

    Returns:
        list: List of sentence strings

    Raises:
        FileNotFoundError: If the sentences file doesn't exist
    """
    print("Loading sentences...")
    with open(filepath, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(sentences)} sentences")
    return sentences


def create_ivf_index(embeddings, nlist=50, nprobe=10):
    """
    Create a FAISS IVF (Inverted File Index) with flat L2 distance.

    This index partitions the vector space into Voronoi cells for faster search.
    The search first identifies which cell(s) the query belongs to, then performs
    exhaustive search only within those cells.

    Args:
        embeddings (np.ndarray): Array of embeddings to index
        nlist (int): Number of partitions (Voronoi cells) to create.
                    Higher values = more partitions = faster search but less accuracy.
                    Default: 50
        nprobe (int): Number of nearby cells to search during query.
                     Higher values = more cells searched = better accuracy but slower.
                     Default: 10

    Returns:
        faiss.IndexIVFFlat: Trained FAISS index ready for searching

    Notes:
        - The index must be trained before adding vectors
        - nlist should be chosen based on dataset size (typically sqrt(n) to n/100)
        - nprobe trades off speed vs accuracy (1 = fastest/least accurate, nlist = slowest/most accurate)
    """
    dimension = embeddings.shape[1]

    # Create quantizer (used to assign vectors to cells)
    quantizer = faiss.IndexFlatL2(dimension)

    # Create IVF index with flat L2 distance
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

    print(f"Index created. Training required: {not index.is_trained}")

    # Train the index to learn the partition structure
    print(f"Training index with {nlist} partitions...")
    index.train(embeddings.astype('float32'))
    print(f"Training complete. Index trained: {index.is_trained}")

    # Add embeddings to the index
    print("Adding embeddings to index...")
    index.add(embeddings.astype('float32'))
    print(f"FAISS IVF index created with {index.ntotal} vectors")

    # Set number of cells to search
    index.nprobe = nprobe
    print(f"Search will probe {nprobe} cells per query")

    # Enable direct map for reconstruction (if needed later)
    index.make_direct_map()

    return index


def perform_search(index, model, query, k=5):
    """
    Perform semantic search using FAISS index.

    Args:
        index (faiss.Index): FAISS index to search in
        model (SentenceTransformer): Model for encoding the query
        query (str): Search query text
        k (int): Number of nearest neighbors to retrieve. Default: 5

    Returns:
        tuple: (distances, indices) where:
            - distances (np.ndarray): L2 distances to nearest neighbors
            - indices (np.ndarray): Indices of nearest neighbors in the original embeddings
    """
    # Encode the query
    query_embedding = model.encode([query]).astype('float32')

    # Search in FAISS index
    distances, indices = index.search(query_embedding, k)

    return distances, indices


def display_results(query, indices, distances, sentences, max_chars=200):
    """
    Display search results in a formatted manner.

    Args:
        query (str): The search query
        indices (np.ndarray): Indices of matched sentences
        distances (np.ndarray): Distances to matched sentences
        sentences (list): List of all sentences
        max_chars (int): Maximum characters to display per result. Default: 200
    """
    print(f"Query: '{query}'")
    print("-" * 80)

    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        if idx < len(sentences):
            sentence_preview = sentences[idx][:max_chars]
            if len(sentences[idx]) > max_chars:
                sentence_preview += "..."
            print(f"{i}. [Distance: {dist:.4f}] {sentence_preview}")
    print("\n")


def main():
    """
    Main function to demonstrate FAISS semantic search with IVF partitioning.

    This function:
    1. Loads pre-computed embeddings and sentences
    2. Creates an IVF partitioned FAISS index
    3. Performs example searches
    4. Displays results
    """
    # Load data
    embeddings = load_embeddings("sentence_embeddings.npy")
    sentences = load_sentences("sentences.txt")

    # Initialize the sentence transformer model
    print("\nInitializing sentence transformer model...")
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Create IVF index with partitioning
    # nlist: number of partitions (Voronoi cells)
    # nprobe: number of cells to search (trade-off between speed and accuracy)
    index = create_ivf_index(embeddings, nlist=50, nprobe=10)

    # Example search queries
    queries = [
        "What are language models?",
        "How do transformers work?",
        "Machine learning techniques",
        "Neural network architecture"
    ]

    # Perform searches
    k = 5  # Number of nearest neighbors to retrieve
    print(f"\n{'='*80}")
    print("SEARCHING FOR SIMILAR SENTENCES")
    print(f"{'='*80}\n")

    for query in queries:
        distances, indices = perform_search(index, model, query, k)
        display_results(query, indices, distances, sentences)

    print(f"{'='*80}")
    print("Search complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


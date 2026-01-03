"""
FAISS Semantic Search with Product Quantization

This module implements semantic search using FAISS with Product Quantization (PQ)
for memory-efficient vector compression and fast similarity search.

Product Quantization (PQ) works by:
1. Splitting each vector into m subvectors
2. Clustering each subvector set independently to create centroids
3. Replacing each subvector with the ID of its nearest centroid
4. This compresses vectors significantly while maintaining search accuracy

The implementation uses:
- IndexIVFPQ: Combines IVF partitioning with Product Quantization
- Sentence transformers for encoding text to embeddings
- Memory-efficient compression for large-scale search

Author: Auto-generated
Date: 2026-01-03
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time


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


def create_flat_index(embeddings):
    """
    Create a basic FAISS Flat L2 index (baseline for comparison).

    Args:
        embeddings (np.ndarray): Array of embeddings to index

    Returns:
        faiss.IndexFlatL2: FAISS flat index (exhaustive search)
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    print(f"Flat L2 index created with {index.ntotal} vectors")
    return index


def create_ivf_index(embeddings, nlist=50, nprobe=10):
    """
    Create a FAISS IVF (Inverted File Index) with flat L2 distance.

    This index partitions the vector space into Voronoi cells for faster search.

    Args:
        embeddings (np.ndarray): Array of embeddings to index
        nlist (int): Number of partitions (Voronoi cells). Default: 50
        nprobe (int): Number of cells to search during query. Default: 10

    Returns:
        faiss.IndexIVFFlat: Trained FAISS IVF index
    """
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)

    print(f"Training IVF index with {nlist} partitions...")
    index.train(embeddings.astype('float32'))
    index.add(embeddings.astype('float32'))
    index.nprobe = nprobe
    print(f"IVF index created with {index.ntotal} vectors")

    return index


def create_ivfpq_index(embeddings, nlist=50, m=8, bits=8, nprobe=10):
    """
    Create a FAISS IndexIVFPQ with Product Quantization for compressed search.

    Product Quantization compresses vectors by:
    1. Splitting each d-dimensional vector into m subvectors (d/m dimensions each)
    2. Clustering each subvector set to create 2^bits centroids per subvector
    3. Replacing subvectors with centroid IDs (only log2(centroids) bits per ID)

    This dramatically reduces memory usage while maintaining search quality.

    Args:
        embeddings (np.ndarray): Array of embeddings to index
        nlist (int): Number of IVF partitions (Voronoi cells). Default: 50
        m (int): Number of subvector segments. Must divide embedding dimension evenly.
                Default: 8 (for 768-dim vectors: 768/8 = 96 dims per subvector)
        bits (int): Number of bits per centroid ID. Creates 2^bits centroids per subvector.
                   Default: 8 (creates 256 centroids per subvector)
        nprobe (int): Number of cells to search during query. Default: 10

    Returns:
        faiss.IndexIVFPQ: Trained FAISS index with Product Quantization

    Notes:
        - Embedding dimension must be divisible by m
        - Each compressed vector uses only m * bits total bits
        - Example: 768-dim vector with m=8, bits=8 uses only 64 bits (8 bytes)
          vs original 768*4=3072 bytes (float32) = 384x compression!
        - Trade-off: Compression comes at cost of slight accuracy loss
    """
    dimension = embeddings.shape[1]

    # Validate that dimension is divisible by m
    if dimension % m != 0:
        raise ValueError(f"Embedding dimension ({dimension}) must be divisible by m ({m})")

    # Create quantizer for IVF partitioning
    quantizer = faiss.IndexFlatL2(dimension)

    # Create IndexIVFPQ: combines IVF partitioning with Product Quantization
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits)

    print(f"\nCreating IndexIVFPQ:")
    print(f"  - IVF partitions (nlist): {nlist}")
    print(f"  - Subvectors (m): {m}")
    print(f"  - Bits per subvector: {bits}")
    print(f"  - Centroids per subvector: {2**bits}")
    print(f"  - Compression ratio: {(dimension * 4 * 8) / (m * bits):.1f}x")

    # Train the index (required for both IVF and PQ)
    print(f"Training index...")
    index.train(embeddings.astype('float32'))
    print(f"Training complete. Index trained: {index.is_trained}")

    # Add embeddings to the index
    print("Adding embeddings to index...")
    index.add(embeddings.astype('float32'))
    print(f"IndexIVFPQ created with {index.ntotal} vectors")

    # Set number of cells to search
    index.nprobe = nprobe
    print(f"Search will probe {nprobe} cells per query\n")

    return index


def perform_search_with_timing(index, model, query, k=5):
    """
    Perform semantic search and measure search time.

    Args:
        index (faiss.Index): FAISS index to search in
        model (SentenceTransformer): Model for encoding the query
        query (str): Search query text
        k (int): Number of nearest neighbors to retrieve. Default: 5

    Returns:
        tuple: (distances, indices, search_time_ms) where:
            - distances (np.ndarray): L2 distances to nearest neighbors
            - indices (np.ndarray): Indices of nearest neighbors
            - search_time_ms (float): Search time in milliseconds
    """
    # Encode the query
    query_embedding = model.encode([query]).astype('float32')

    # Time the search
    start_time = time.time()
    distances, indices = index.search(query_embedding, k)
    search_time = (time.time() - start_time) * 1000  # Convert to ms

    return distances, indices, search_time


def display_results(query, indices, distances, sentences, search_time, max_chars=150):
    """
    Display search results in a formatted manner.

    Args:
        query (str): The search query
        indices (np.ndarray): Indices of matched sentences
        distances (np.ndarray): Distances to matched sentences
        sentences (list): List of all sentences
        search_time (float): Search time in milliseconds
        max_chars (int): Maximum characters to display per result. Default: 150
    """
    print(f"Query: '{query}' [Search time: {search_time:.3f}ms]")
    print("-" * 80)

    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        if idx < len(sentences) and idx >= 0:  # Check valid index
            sentence_preview = sentences[idx][:max_chars]
            if len(sentences[idx]) > max_chars:
                sentence_preview += "..."
            print(f"{i}. [Distance: {dist:.4f}] {sentence_preview}")
    print()


def compare_index_performance(embeddings, sentences, model, queries, k=5):
    """
    Compare performance of different FAISS index types.

    Args:
        embeddings (np.ndarray): Array of embeddings
        sentences (list): List of sentences
        model (SentenceTransformer): Sentence transformer model
        queries (list): List of query strings to test
        k (int): Number of results to retrieve. Default: 5

    Returns:
        dict: Performance metrics for each index type
    """
    print("="*80)
    print("COMPARING FAISS INDEX TYPES")
    print("="*80)

    results = {}

    # 1. Flat L2 Index (Baseline)
    print("\n1. FLAT L2 INDEX (Baseline - Exhaustive Search)")
    print("-" * 80)
    flat_index = create_flat_index(embeddings)
    flat_times = []

    for query in queries:
        distances, indices, search_time = perform_search_with_timing(flat_index, model, query, k)
        display_results(query, indices, distances, sentences, search_time)
        flat_times.append(search_time)

    results['Flat L2'] = {
        'avg_time_ms': np.mean(flat_times),
        'memory': 'Full vectors (no compression)'
    }

    # 2. IVF Index
    print("\n2. IVF INDEX (Partitioned Search)")
    print("-" * 80)
    ivf_index = create_ivf_index(embeddings, nlist=50, nprobe=10)
    ivf_times = []

    for query in queries:
        distances, indices, search_time = perform_search_with_timing(ivf_index, model, query, k)
        display_results(query, indices, distances, sentences, search_time)
        ivf_times.append(search_time)

    results['IVF'] = {
        'avg_time_ms': np.mean(ivf_times),
        'memory': 'Full vectors (no compression)'
    }

    # 3. IVFPQ Index (with Product Quantization)
    print("\n3. IVFPQ INDEX (Partitioned + Product Quantization)")
    print("-" * 80)
    ivfpq_index = create_ivfpq_index(embeddings, nlist=50, m=8, bits=8, nprobe=10)
    ivfpq_times = []

    for query in queries:
        distances, indices, search_time = perform_search_with_timing(ivfpq_index, model, query, k)
        display_results(query, indices, distances, sentences, search_time)
        ivfpq_times.append(search_time)

    dimension = embeddings.shape[1]
    compression_ratio = (dimension * 4 * 8) / (8 * 8)  # m=8, bits=8
    results['IVFPQ'] = {
        'avg_time_ms': np.mean(ivfpq_times),
        'memory': f'Compressed {compression_ratio:.1f}x'
    }

    # Print comparison summary
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Index Type':<15} {'Avg Search Time':<20} {'Memory Usage':<30}")
    print("-" * 80)

    for index_type, metrics in results.items():
        print(f"{index_type:<15} {metrics['avg_time_ms']:>10.3f} ms      {metrics['memory']:<30}")

    print("\nSpeedup vs Flat L2:")
    for index_type, metrics in results.items():
        if index_type != 'Flat L2':
            speedup = results['Flat L2']['avg_time_ms'] / metrics['avg_time_ms']
            print(f"  {index_type}: {speedup:.2f}x faster")

    print("="*80)

    return results


def main():
    """
    Main function to demonstrate FAISS Product Quantization.

    This function:
    1. Loads pre-computed embeddings and sentences
    2. Creates and compares three index types:
       - Flat L2 (baseline, exhaustive search)
       - IVF (partitioned search)
       - IVFPQ (partitioned + compressed search)
    3. Demonstrates the trade-offs between speed, memory, and accuracy
    """
    # Load data
    embeddings = load_embeddings("sentence_embeddings.npy")
    sentences = load_sentences("sentences.txt")

    # Initialize the sentence transformer model
    print("\nInitializing sentence transformer model...")
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    print("Model loaded successfully!\n")

    # Example search queries
    queries = [
        "What are language models?",
        "How do transformers work?",
        "Machine learning techniques"
    ]

    # Compare performance of different index types
    k = 5  # Number of nearest neighbors to retrieve
    results = compare_index_performance(embeddings, sentences, model, queries, k)

    print("\nKey Takeaways:")
    print("-" * 80)
    print("• Flat L2: Perfect accuracy, slowest, high memory usage")
    print("• IVF: Good accuracy, faster, same memory as Flat")
    print("• IVFPQ: Slight accuracy loss, fastest, dramatically reduced memory")
    print("\nFor large-scale applications (millions of vectors), IVFPQ is essential!")
    print("="*80)


if __name__ == "__main__":
    main()


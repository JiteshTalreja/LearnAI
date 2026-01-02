import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the saved embeddings
print("Loading embeddings...")
embeddings = np.load("sentence_embeddings.npy")
print(f"Loaded embeddings with shape: {embeddings.shape}")

# Load the corresponding sentences
print("Loading sentences...")
with open("sentences.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines()]
print(f"Loaded {len(sentences)} sentences")

# Initialize the same model for encoding queries
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Create FAISS index
dimension = embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(dimension)  # L2 distance index
index.add(embeddings.astype('float32'))  # Add embeddings to index
print(f"FAISS index created with {index.ntotal} vectors")

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
    print(f"Query: '{query}'")
    print("-" * 80)

    # Encode the query
    query_embedding = model.encode([query]).astype('float32')

    # Search in FAISS index
    distances, indices = index.search(query_embedding, k)

    # Display results
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        if idx < len(sentences):
            print(f"{i}. [Distance: {dist:.4f}] {sentences[idx][:200]}...")
    print("\n")

print(f"{'='*80}")
print("Search complete!")
print(f"{'='*80}")

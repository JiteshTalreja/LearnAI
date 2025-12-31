"""
Sentence-BERT (SBERT) Semantic Similarity - Manual Implementation.

Sentence-BERT is a modification of BERT that produces semantically meaningful sentence embeddings.
Instead of using individual token embeddings, SBERT creates fixed-size sentence vectors by pooling
the output of transformer models, making them suitable for similarity comparisons.

This implementation shows the core concepts:
1. Load a pre-trained transformer model (e.g., BERT, DistilBERT)
2. Tokenize input sentences
3. Pass tokens through the transformer to get contextual embeddings
4. Apply pooling (mean pooling) to get a single sentence vector
5. Use cosine similarity to compare sentence vectors

Key Concepts:
- **Transformer Encoding**: Each word gets a contextual embedding based on surrounding words
- **Mean Pooling**: Average all token embeddings to create one sentence vector
- **Cosine Similarity**: Measures angle between vectors (semantic similarity)

Mathematical Foundation:

Given sentence S with tokens [t1, t2, ..., tn]:
1. Transformer outputs token embeddings: [e1, e2, ..., en]
2. Mean pooling: sentence_embedding = (e1 + e2 + ... + en) / n
3. Cosine similarity between sentences A and B:
   similarity = (A · B) / (||A|| × ||B||)

Properties:
- **Dense Vectors**: Typically 384-768 dimensions (depends on model)
- **Semantic Understanding**: Captures meaning beyond exact word matches
- **Context-Aware**: Same word has different embeddings in different contexts
- **Transfer Learning**: Uses pre-trained models fine-tuned on semantic tasks

Applications:
- Semantic search and document retrieval
- Paraphrase detection
- Question answering
- Duplicate detection
- Text clustering
- Recommendation systems

Example:
Query: "How do I learn Python?"
Doc1: "Python tutorial for beginners"
Doc2: "Java programming guide"

SBERT would recognize that Doc1 is semantically closer to the query even though
they share no exact words, whereas keyword-based methods might fail.

Time Complexity: O(n × L × d²) where n = documents, L = sequence length, d = hidden dimension
Space Complexity: O(n × d) for storing embeddings

"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import List, Tuple, Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Note: torch and transformers not installed. Install with: pip install torch transformers")


class SimpleSBERT:
    """
    Simplified Sentence-BERT implementation using transformers directly.
    Uses mean pooling over BERT token embeddings to create sentence vectors.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", device: Optional[str] = None):
        """
        Initialize the SBERT model.

        Args:
            model_name: HuggingFace model name (default: distilbert-base-uncased)
                       - distilbert-base-uncased: Fast, 66M params, 768-dim
                       - bert-base-uncased: Standard BERT, 110M params, 768-dim
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("torch and transformers required. Install with: pip install torch transformers")

        self.model_name = model_name

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Loading model: {model_name} on {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Apply mean pooling to token embeddings.

        This averages all token embeddings, weighted by the attention mask to ignore padding.

        Args:
            token_embeddings: Tensor of shape (batch_size, seq_len, hidden_dim)
            attention_mask: Tensor of shape (batch_size, seq_len)

        Returns:
            Pooled embeddings of shape (batch_size, hidden_dim)
        """
        # Expand attention mask to match token embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings, weighted by mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # Sum mask values (to get actual token count, excluding padding)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # Calculate mean
        return sum_embeddings / sum_mask

    def encode(self, sentences: List[str], batch_size: int = 8,
               normalize: bool = True, show_progress: bool = True) -> np.ndarray:
        """
        Encode sentences into embeddings.

        Args:
            sentences: List of sentences to encode
            batch_size: Number of sentences to process at once
            normalize: Whether to L2-normalize the embeddings
            show_progress: Whether to print progress

        Returns:
            NumPy array of shape (n_sentences, embedding_dim)
        """
        if not sentences:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        all_embeddings = []

        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            if show_progress:
                print(f"  Encoding batch {i // batch_size + 1}/{(len(sentences) - 1) // batch_size + 1}")

            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get embeddings
            with torch.no_grad():
                model_output = self.model(**encoded)

                # Get token embeddings (last hidden state)
                token_embeddings = model_output.last_hidden_state

                # Apply mean pooling
                sentence_embeddings = self.mean_pooling(token_embeddings, encoded['attention_mask'])

                # Normalize if requested
                if normalize:
                    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

                # Move to CPU and convert to numpy
                embeddings = sentence_embeddings.cpu().numpy()
                all_embeddings.append(embeddings)

        # Concatenate all batches
        return np.vstack(all_embeddings).astype(np.float32)

    def encode_single(self, sentence: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single sentence.

        Args:
            sentence: Sentence to encode
            normalize: Whether to L2-normalize the embedding

        Returns:
            NumPy array of shape (embedding_dim,)
        """
        return self.encode([sentence], batch_size=1, normalize=normalize, show_progress=False)[0]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (between -1 and 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for all embeddings.

    Args:
        embeddings: NumPy array of shape (n_documents, embedding_dim)

    Returns:
        Similarity matrix of shape (n_documents, n_documents)
    """
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms

    # Compute similarity matrix
    return normalized @ normalized.T


def rank_documents(query_embedding: np.ndarray, document_embeddings: np.ndarray,
                   top_k: int = 5) -> List[Tuple[int, float]]:
    """
    Rank documents by similarity to query.

    Args:
        query_embedding: Query vector of shape (embedding_dim,)
        document_embeddings: Document vectors of shape (n_documents, embedding_dim)
        top_k: Number of top documents to return

    Returns:
        List of (document_index, similarity_score) tuples
    """
    if document_embeddings.size == 0:
        return []

    # Compute similarities
    similarities = []
    for i, doc_emb in enumerate(document_embeddings):
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append((i, sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def semantic_search(query: str, documents: List[str], embedder: SimpleSBERT,
                   document_embeddings: Optional[np.ndarray] = None,
                   top_k: int = 3) -> List[Tuple[int, float, str]]:
    """
    Perform semantic search: find most similar documents to query.

    Args:
        query: Query string
        documents: List of document strings
        embedder: SimpleSBERT instance
        document_embeddings: Pre-computed document embeddings (optional)
        top_k: Number of top results to return

    Returns:
        List of (index, score, document_text) tuples
    """
    # Encode documents if not provided
    if document_embeddings is None:
        print("Encoding documents...")
        document_embeddings = embedder.encode(documents, show_progress=False)

    # Encode query
    query_embedding = embedder.encode_single(query)

    # Rank documents
    rankings = rank_documents(query_embedding, document_embeddings, top_k=top_k)

    # Return with document text
    return [(idx, score, documents[idx]) for idx, score in rankings]


# Example Usage
if __name__ == "__main__":
    print("=" * 80)
    print("SIMPLE SBERT IMPLEMENTATION - SEMANTIC SEARCH DEMO")
    print("=" * 80)
    print()

    # Sample documents
    documents = [
        "The cat sat on the mat",
        "The dog sat on the log",
        "Cats and dogs are animals",
        "Python is a programming language",
        "Machine learning uses Python for data analysis",
        "Natural language processing is part of machine learning",
        "Deep learning is a subset of machine learning",
        "Neural networks are used in deep learning",
        "Python programming is popular for machine learning",
        "Data science requires statistics and programming"
    ]

    print(f"Documents ({len(documents)}):")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    print()

    # Initialize SBERT
    print("Initializing SimpleSBERT...")
    print("Note: First run will download the model (~250MB)")
    print()

    try:
        embedder = SimpleSBERT(model_name="distilbert-base-uncased")
    except Exception as e:
        print(f"Error: {e}")
        print("Install: pip install torch transformers")
        exit(1)

    print()
    print("=" * 80)
    print("ENCODING DOCUMENTS")
    print("=" * 80)

    # Encode all documents
    doc_embeddings = embedder.encode(documents, batch_size=4, show_progress=True)
    print(f"\nEncoded {len(documents)} documents → shape: {doc_embeddings.shape}")

    print()
    print("=" * 80)
    print("DOCUMENT SIMILARITY MATRIX")
    print("=" * 80)

    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(doc_embeddings)

    print(f"\nSimilarity between selected document pairs:")
    print(f"  Doc 1 & Doc 2 (cat/dog): {sim_matrix[0, 1]:.4f}")
    print(f"  Doc 5 & Doc 7 (ML topics): {sim_matrix[4, 6]:.4f}")
    print(f"  Doc 1 & Doc 5 (unrelated): {sim_matrix[0, 4]:.4f}")

    print()
    print("=" * 80)
    print("SEMANTIC QUERY SEARCH")
    print("=" * 80)

    # Test queries
    queries = [
        "machine learning with python",
        "pets sitting on furniture",
        "introduction to neural networks",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = semantic_search(query, documents, embedder, doc_embeddings, top_k=3)

        print("Top 3 Results:")
        for rank, (idx, score, text) in enumerate(results, 1):
            print(f"  {rank}. [Score: {score:.4f}] {text}")

    print()
    print("=" * 80)
    print("SEMANTIC vs KEYWORD MATCHING")
    print("=" * 80)

    query = "How to learn coding?"
    print(f"\nQuery: '{query}'")
    print("(Note: No exact word overlap with most documents)")

    results = semantic_search(query, documents, embedder, doc_embeddings, top_k=3)

    print("\nSBERT Results (semantic understanding):")
    for rank, (idx, score, text) in enumerate(results, 1):
        print(f"  {rank}. [Score: {score:.4f}] {text}")

    print()
    print("=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
SBERT captures semantic meaning beyond exact keyword matches:
- TF-IDF/BM25: Fast, keyword-based, good for exact matches
- SBERT: Semantic understanding, slower but more intelligent

Best for: Semantic search, paraphrase detection, Q&A, clustering
    """)


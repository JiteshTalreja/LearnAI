"""
TF-IDF (Term Frequency-Inverse Document Frequency) Implementation using NumPy.

TF-IDF is a numerical statistic that is intended to reflect how important a word is to a document
in a collection or corpus. It is one of the most popular term-weighting schemes and is widely used
in information retrieval and text mining.

TF-IDF is composed of two main components:

1. Term Frequency (TF): Measures how frequently a term occurs in a document.
   TF(t, d) = (Number of times term t appears in document d) / (Total number of terms in document d)

   Variations:
   - Raw count: Just the frequency of the term
   - Boolean: 1 if term present, 0 otherwise
   - Log normalization: 1 + log(frequency)
   - Augmented frequency: 0.5 + 0.5 * (frequency / max_frequency_in_doc)

2. Inverse Document Frequency (IDF): Measures how important a term is across the corpus.
   IDF(t, D) = log(Total number of documents / Number of documents containing term t)

   The IDF diminishes the weight of terms that occur very frequently across the corpus and
   increases the weight of terms that occur rarely.

Final TF-IDF Score:
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

Key Properties:
1. Range: TF-IDF scores are typically positive real numbers
2. High scores: Indicate terms that are frequent in a document but rare across the corpus
3. Low scores: Indicate terms that are common across many documents (e.g., "the", "is", "and")
4. Zero scores: Terms that don't appear in the document

Applications:
- Search engines and information retrieval
- Document classification and clustering
- Text summarization
- Keyword extraction
- Recommendation systems
- Spam filtering
- Content-based filtering

Example: Consider a corpus of 3 documents:
Doc1: "the cat sat on the mat"
Doc2: "the dog sat on the log"
Doc3: "cats and dogs are animals"

For the term "cat" in Doc1:
- TF = 1/6 = 0.167 (appears once, document has 6 words)
- IDF = log(3/1) = 1.099 (3 total docs, "cat" appears in 1)
- TF-IDF = 0.167 × 1.099 = 0.183

Time Complexity: O(n * m) where n is the number of documents and m is the vocabulary size
Space Complexity: O(n * m) for storing the TF-IDF matrix

"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter
import re


class TFIDFVectorizer:
    """
    A class to compute TF-IDF vectors for a collection of documents.
    """

    def __init__(self, use_idf: bool = True, smooth_idf: bool = True,
                 sublinear_tf: bool = False, max_features: int = None,
                 lowercase: bool = True, stop_words: Set[str] = None):
        """
        Initialize the TF-IDF vectorizer.

        Args:
            use_idf: Enable inverse-document-frequency reweighting
            smooth_idf: Add one to document frequencies (prevents zero divisions)
            sublinear_tf: Apply sublinear tf scaling (1 + log(tf))
            max_features: Maximum number of features (top terms by frequency)
            lowercase: Convert all characters to lowercase
            stop_words: Set of words to ignore
        """
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.max_features = max_features
        self.lowercase = lowercase
        self.stop_words = stop_words or set()

        self.vocabulary_ = {}
        self.idf_ = None
        self.n_documents_ = 0

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text string

        Returns:
            List of tokens
        """
        if self.lowercase:
            text = text.lower()
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text)
        # Filter stop words
        tokens = [t for t in tokens if t not in self.stop_words]
        return tokens

    def build_vocabulary(self, documents: List[str]) -> None:
        """
        Build vocabulary from documents.

        Args:
            documents: List of document strings
        """
        # Count term frequencies across all documents
        term_doc_count = Counter()

        for doc in documents:
            tokens = self.tokenize(doc)
            unique_tokens = set(tokens)
            term_doc_count.update(unique_tokens)

        # Select features
        if self.max_features:
            # Keep only top max_features terms
            most_common = term_doc_count.most_common(self.max_features)
            terms = [term for term, _ in most_common]
        else:
            terms = sorted(term_doc_count.keys())

        # Build vocabulary mapping
        self.vocabulary_ = {term: idx for idx, term in enumerate(terms)}

    def compute_tf(self, document: str) -> np.ndarray:
        """
        Compute term frequency vector for a document.

        Args:
            document: Document string

        Returns:
            NumPy array of term frequencies
        """
        tokens = self.tokenize(document)
        tf_vector = np.zeros(len(self.vocabulary_))

        if len(tokens) == 0:
            return tf_vector

        token_counts = Counter(tokens)

        for term, count in token_counts.items():
            if term in self.vocabulary_:
                idx = self.vocabulary_[term]

                if self.sublinear_tf:
                    # Sublinear TF scaling: 1 + log(count)
                    tf_vector[idx] = 1 + np.log(count)
                else:
                    # Normalized term frequency
                    tf_vector[idx] = count / len(tokens)

        return tf_vector

    def compute_idf(self, documents: List[str]) -> np.ndarray:
        """
        Compute inverse document frequency for all terms.

        Args:
            documents: List of document strings

        Returns:
            NumPy array of IDF values
        """
        n_docs = len(documents)
        doc_freq = np.zeros(len(self.vocabulary_))

        # Count document frequency for each term
        for doc in documents:
            tokens = self.tokenize(doc)
            unique_tokens = set(tokens)

            for term in unique_tokens:
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    doc_freq[idx] += 1

        # Compute IDF
        if self.smooth_idf:
            # Add 1 to avoid division by zero
            idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1
        else:
            # Standard IDF
            idf = np.log(n_docs / (doc_freq + 1e-10)) + 1

        return idf

    def fit(self, documents: List[str]) -> 'TFIDFVectorizer':
        """
        Learn vocabulary and IDF from training documents.

        Args:
            documents: List of document strings

        Returns:
            Self (for method chaining)
        """
        self.n_documents_ = len(documents)

        # Build vocabulary
        self.build_vocabulary(documents)

        # Compute IDF if enabled
        if self.use_idf:
            self.idf_ = self.compute_idf(documents)
        else:
            self.idf_ = np.ones(len(self.vocabulary_))

        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to TF-IDF vectors.

        Args:
            documents: List of document strings

        Returns:
            NumPy array of shape (n_documents, n_features)
        """
        if not self.vocabulary_:
            raise ValueError("Vocabulary not built. Call fit() first.")

        tfidf_matrix = np.zeros((len(documents), len(self.vocabulary_)))

        for i, doc in enumerate(documents):
            tf_vector = self.compute_tf(doc)
            tfidf_matrix[i] = tf_vector * self.idf_

        return tfidf_matrix

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        """
        Learn vocabulary and IDF, then transform documents to TF-IDF vectors.

        Args:
            documents: List of document strings

        Returns:
            NumPy array of shape (n_documents, n_features)
        """
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names (terms in vocabulary).

        Returns:
            List of feature names
        """
        sorted_vocab = sorted(self.vocabulary_.items(), key=lambda x: x[1])
        return [term for term, _ in sorted_vocab]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (between -1 and 1, typically 0 to 1 for TF-IDF)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def compute_similarity_matrix(tfidf_matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for all documents.

    Args:
        tfidf_matrix: TF-IDF matrix of shape (n_documents, n_features)

    Returns:
        Similarity matrix of shape (n_documents, n_documents)
    """
    # Normalize vectors
    norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_matrix = tfidf_matrix / norms

    # Compute cosine similarity
    similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

    return similarity_matrix


def get_top_terms(tfidf_vector: np.ndarray, feature_names: List[str],
                  top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Get top N terms by TF-IDF score for a document.

    Args:
        tfidf_vector: TF-IDF vector for a document
        feature_names: List of feature names
        top_n: Number of top terms to return

    Returns:
        List of (term, score) tuples
    """
    # Get indices of top scores
    top_indices = np.argsort(tfidf_vector)[::-1][:top_n]

    # Get terms and scores
    top_terms = [(feature_names[i], tfidf_vector[i]) for i in top_indices
                 if tfidf_vector[i] > 0]

    return top_terms


# Example Usage
if __name__ == "__main__":
    print("=" * 80)
    print("TF-IDF VECTORIZATION EXAMPLE")
    print("=" * 80)

    # Sample documents
    documents = [
        "The cat sat on the mat",
        "The dog sat on the log",
        "Cats and dogs are animals",
        "Python is a programming language",
        "Machine learning uses Python for data analysis",
        "Natural language processing is part of machine learning"
    ]

    print("\nDocuments:")
    for i, doc in enumerate(documents):
        print(f"{i+1}. {doc}")

    # Create and fit vectorizer
    vectorizer = TFIDFVectorizer(
        use_idf=True,
        smooth_idf=True,
        lowercase=True,
        stop_words={'the', 'is', 'on', 'a', 'and', 'are', 'of', 'for'}
    )

    # Compute TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names()

    print(f"\nVocabulary size: {len(feature_names)}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    print("\n" + "=" * 80)
    print("TOP TERMS PER DOCUMENT")
    print("=" * 80)

    # Get top terms for each document
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}: '{doc}'")
        top_terms = get_top_terms(tfidf_matrix[i], feature_names, top_n=5)
        print("Top terms:")
        for term, score in top_terms:
            print(f"  - {term}: {score:.4f}")

    print("\n" + "=" * 80)
    print("DOCUMENT SIMILARITY ANALYSIS")
    print("=" * 80)

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(tfidf_matrix)

    print("\nPairwise Cosine Similarity Matrix:")
    print(np.round(similarity_matrix, 3))

    # Find most similar documents
    print("\nMost Similar Document Pairs:")
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            sim = similarity_matrix[i, j]
            if sim > 0.1:  # Only show pairs with similarity > 0.1
                print(f"\nDoc {i+1} & Doc {j+1}: {sim:.4f}")
                print(f"  Doc {i+1}: {documents[i]}")
                print(f"  Doc {j+1}: {documents[j]}")

    print("\n" + "=" * 80)
    print("QUERY SEARCH EXAMPLE")
    print("=" * 80)

    # Query example
    query = "machine learning with Python"
    print(f"\nQuery: '{query}'")

    # Transform query to TF-IDF vector
    query_vector = vectorizer.transform([query])[0]

    # Find most similar documents
    similarities = []
    for i in range(len(documents)):
        sim = cosine_similarity(query_vector, tfidf_matrix[i])
        similarities.append((i, sim))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 3 Most Relevant Documents:")
    for rank, (doc_idx, sim) in enumerate(similarities[:3], 1):
        print(f"{rank}. [{sim:.4f}] {documents[doc_idx]}")

    print("\n" + "=" * 80)
    print("SUBLINEAR TF SCALING COMPARISON")
    print("=" * 80)

    # Compare standard vs sublinear TF
    test_doc = "python python python data analysis"
    print(f"\nTest document: '{test_doc}'")

    vectorizer_standard = TFIDFVectorizer(use_idf=False, sublinear_tf=False)
    vectorizer_sublinear = TFIDFVectorizer(use_idf=False, sublinear_tf=True)

    vectorizer_standard.fit([test_doc])
    vectorizer_sublinear.fit([test_doc])

    tf_standard = vectorizer_standard.transform([test_doc])[0]
    tf_sublinear = vectorizer_sublinear.transform([test_doc])[0]

    print("\nTerm Frequencies (Standard vs Sublinear):")
    features = vectorizer_standard.get_feature_names()
    for i, term in enumerate(features):
        if tf_standard[i] > 0:
            print(f"  {term}: {tf_standard[i]:.4f} (standard) vs {tf_sublinear[i]:.4f} (sublinear)")


"""
BM25 (Best Matching 25) Implementation using NumPy.

BM25 is a ranking function used in information retrieval to estimate the relevance of documents
to a given search query. It is based on the probabilistic retrieval framework and is considered
a state-of-the-art TF-IDF-like retrieval function.

BM25 improves upon TF-IDF by:
1. Using a saturation function for term frequency (prevents over-weighting of term frequency)
2. Considering document length normalization (penalizes very long documents)
3. Providing tunable parameters for customization

BM25 Formula:
BM25(q, d) = Σ IDF(qi) × [f(qi, d) × (k1 + 1)] / [f(qi, d) + k1 × (1 - b + b × |d| / avgdl)]

Where:
- q is the query containing keywords q1, ..., qn
- d is a document
- f(qi, d) is the frequency of term qi in document d
- |d| is the length of document d (in words)
- avgdl is the average document length in the corpus
- k1 is a parameter that controls term frequency saturation (typically 1.2 to 2.0)
- b is a parameter that controls document length normalization (typically 0.75)
- IDF(qi) is the inverse document frequency of term qi

IDF Component (similar to TF-IDF but slightly different):
IDF(qi) = log[(N - n(qi) + 0.5) / (n(qi) + 0.5) + 1]

Where:
- N is the total number of documents in the corpus
- n(qi) is the number of documents containing term qi

Key Parameters:
1. k1 (term frequency saturation parameter):
   - Controls how quickly the score saturates with respect to term frequency
   - Higher k1 means term frequency has more impact
   - Typical range: 1.2 to 2.0
   - Default: 1.5

2. b (length normalization parameter):
   - Controls how much document length affects the score
   - b = 0: no length normalization
   - b = 1: full length normalization
   - Typical value: 0.75
   - Default: 0.75

Properties:
1. Non-linear term frequency: Unlike TF-IDF, BM25 uses a saturation function
2. Document length normalization: Longer documents are penalized
3. Tunable parameters: Can be optimized for specific domains
4. Probabilistic foundation: Based on probabilistic retrieval models

Advantages over TF-IDF:
- Better handling of term frequency (saturation prevents over-weighting)
- Document length normalization (fairness across documents of different lengths)
- Tunable parameters (can be optimized for specific use cases)
- Generally better ranking performance in information retrieval tasks

Applications:
- Search engines (e.g., Elasticsearch uses BM25 as default)
- Question answering systems
- Document ranking and retrieval
- Recommendation systems
- Information extraction
- Text mining

Example:
Query: "machine learning"
Document: "machine learning is a subset of artificial intelligence that focuses on machine algorithms"

Term frequencies: machine=2, learning=1
Document length: 13 words
Average doc length: 10 words (assuming)

BM25 score calculation involves:
1. IDF for "machine" and "learning"
2. Saturation function applied to term frequencies
3. Document length normalization

Time Complexity: O(n × m) where n is the number of documents and m is the average query length
Space Complexity: O(n × v) where v is the vocabulary size

"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import Counter
import re


class BM25:
    """
    BM25 ranking function implementation.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25,
                 lowercase: bool = True, stop_words: Set[str] = None):
        """
        Initialize BM25 ranker.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
            epsilon: Floor value for IDF (default: 0.25)
            lowercase: Convert all text to lowercase
            stop_words: Set of words to ignore
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.lowercase = lowercase
        self.stop_words = stop_words or set()

        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenized_corpus = []

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
        tokens = re.findall(r'\b\w+\b', text)
        tokens = [t for t in tokens if t not in self.stop_words]
        return tokens

    def fit(self, corpus: List[str]) -> 'BM25':
        """
        Fit the BM25 model on a corpus of documents.

        Args:
            corpus: List of document strings

        Returns:
            Self (for method chaining)
        """
        self.corpus_size = len(corpus)

        # Tokenize all documents
        self.tokenized_corpus = [self.tokenize(doc) for doc in corpus]

        # Calculate document lengths
        self.doc_len = [len(doc) for doc in self.tokenized_corpus]

        # Calculate average document length
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0

        # Calculate document frequencies
        df = {}
        for doc in self.tokenized_corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] = df.get(term, 0) + 1

        # Calculate IDF for each term
        self.idf = {}
        for term, freq in df.items():
            # BM25 IDF formula
            idf_score = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1)
            # Apply epsilon floor
            self.idf[term] = max(idf_score, self.epsilon)

        return self

    def get_scores(self, query: str) -> np.ndarray:
        """
        Calculate BM25 scores for all documents given a query.

        Args:
            query: Query string

        Returns:
            NumPy array of BM25 scores for each document
        """
        query_tokens = self.tokenize(query)
        scores = np.zeros(self.corpus_size)

        # Calculate query term frequencies
        query_term_freqs = Counter(query_tokens)

        for i, doc in enumerate(self.tokenized_corpus):
            doc_term_freqs = Counter(doc)
            doc_score = 0.0

            for term in query_term_freqs:
                if term not in self.idf:
                    continue

                # Term frequency in document
                f = doc_term_freqs.get(term, 0)

                # BM25 score formula
                idf = self.idf[term]
                numerator = f * (self.k1 + 1)
                denominator = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)

                doc_score += idf * (numerator / denominator)

            scores[i] = doc_score

        return scores

    def get_top_n(self, query: str, n: int = 5) -> List[Tuple[int, float]]:
        """
        Get top N documents for a query.

        Args:
            query: Query string
            n: Number of top documents to return

        Returns:
            List of (document_index, score) tuples
        """
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:n]
        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]

    def get_batch_scores(self, queries: List[str]) -> np.ndarray:
        """
        Calculate BM25 scores for multiple queries.

        Args:
            queries: List of query strings

        Returns:
            NumPy array of shape (n_queries, n_documents)
        """
        scores_matrix = np.zeros((len(queries), self.corpus_size))

        for i, query in enumerate(queries):
            scores_matrix[i] = self.get_scores(query)

        return scores_matrix


def compare_bm25_tfidf(corpus: List[str], query: str, k1: float = 1.5, b: float = 0.75):
    """
    Compare BM25 and TF-IDF scoring for a given query and corpus.

    Args:
        corpus: List of document strings
        query: Query string
        k1: BM25 k1 parameter
        b: BM25 b parameter

    Returns:
        Tuple of (bm25_scores, tfidf_scores)
    """
    # BM25 scores
    bm25 = BM25(k1=k1, b=b)
    bm25.fit(corpus)
    bm25_scores = bm25.get_scores(query)

    # Simple TF-IDF scores (for comparison)
    from collections import Counter

    query_tokens = bm25.tokenize(query)
    tfidf_scores = np.zeros(len(corpus))

    for i, doc in enumerate(corpus):
        doc_tokens = bm25.tokenize(doc)
        doc_term_freqs = Counter(doc_tokens)

        for term in query_tokens:
            if term in doc_term_freqs:
                # Simple TF-IDF: TF * IDF
                tf = doc_term_freqs[term] / len(doc_tokens)
                idf = bm25.idf.get(term, 0)
                tfidf_scores[i] += tf * idf

    return bm25_scores, tfidf_scores


# Example Usage
if __name__ == "__main__":
    print("=" * 80)
    print("BM25 RANKING EXAMPLE")
    print("=" * 80)

    # Sample corpus
    corpus = [
        "The cat sat on the mat",
        "The dog sat on the log",
        "Cats and dogs are animals",
        "Python is a programming language",
        "Machine learning uses Python for data analysis",
        "Natural language processing is part of machine learning",
        "Deep learning is a subset of machine learning",
        "Neural networks are used in deep learning",
        "Python programming is popular for machine learning",
        "Data science requires knowledge of statistics and programming"
    ]

    print("\nCorpus Documents:")
    for i, doc in enumerate(corpus):
        print(f"{i+1}. {doc}")

    # Initialize and fit BM25
    bm25 = BM25(
        k1=1.5,
        b=0.75,
        stop_words={'the', 'is', 'on', 'a', 'and', 'are', 'of', 'for'}
    )
    bm25.fit(corpus)

    print(f"\nCorpus Statistics:")
    print(f"  Total documents: {bm25.corpus_size}")
    print(f"  Average document length: {bm25.avgdl:.2f} words")
    print(f"  Vocabulary size: {len(bm25.idf)}")

    print("\n" + "=" * 80)
    print("QUERY RANKING")
    print("=" * 80)

    # Example queries
    queries = [
        "machine learning",
        "Python programming",
        "cats and dogs",
        "deep neural networks"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        top_docs = bm25.get_top_n(query, n=3)

        print("Top 3 Results:")
        for rank, (doc_idx, score) in enumerate(top_docs, 1):
            print(f"  {rank}. [Score: {score:.4f}] {corpus[doc_idx]}")

    print("\n" + "=" * 80)
    print("BM25 vs TF-IDF COMPARISON")
    print("=" * 80)

    query = "machine learning Python"
    print(f"\nQuery: '{query}'")

    bm25_scores, tfidf_scores = compare_bm25_tfidf(corpus, query)

    print("\nDocument Scores Comparison:")
    print(f"{'Doc':<4} {'BM25':<10} {'TF-IDF':<10} {'Document'}")
    print("-" * 80)

    for i in range(len(corpus)):
        print(f"{i+1:<4} {bm25_scores[i]:<10.4f} {tfidf_scores[i]:<10.4f} {corpus[i][:50]}")

    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    query = "machine learning"
    print(f"\nQuery: '{query}'")

    # Test different k1 values
    print("\nEffect of k1 parameter (term frequency saturation):")
    for k1_val in [0.5, 1.2, 1.5, 2.0, 3.0]:
        bm25_temp = BM25(k1=k1_val, b=0.75)
        bm25_temp.fit(corpus)
        scores = bm25_temp.get_scores(query)
        top_score = np.max(scores)
        top_idx = np.argmax(scores)
        print(f"  k1={k1_val:.1f}: Top score={top_score:.4f} for '{corpus[top_idx][:50]}'")

    # Test different b values
    print("\nEffect of b parameter (length normalization):")
    for b_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        bm25_temp = BM25(k1=1.5, b=b_val)
        bm25_temp.fit(corpus)
        scores = bm25_temp.get_scores(query)
        top_score = np.max(scores)
        top_idx = np.argmax(scores)
        print(f"  b={b_val:.2f}: Top score={top_score:.4f} for '{corpus[top_idx][:50]}'")

    print("\n" + "=" * 80)
    print("BATCH QUERY PROCESSING")
    print("=" * 80)

    batch_queries = [
        "Python programming",
        "machine learning",
        "data analysis"
    ]

    print("\nProcessing multiple queries:")
    batch_scores = bm25.get_batch_scores(batch_queries)

    for i, query in enumerate(batch_queries):
        print(f"\nQuery {i+1}: '{query}'")
        top_idx = np.argmax(batch_scores[i])
        top_score = batch_scores[i][top_idx]
        print(f"  Top result: [Score: {top_score:.4f}] {corpus[top_idx]}")

    print("\n" + "=" * 80)
    print("DOCUMENT LENGTH IMPACT")
    print("=" * 80)

    # Create documents with varying lengths
    short_doc = "Python"
    medium_doc = "Python is a programming language"
    long_doc = "Python is a popular programming language widely used in data science machine learning web development and many other domains"

    test_corpus = [short_doc, medium_doc, long_doc]
    test_query = "Python programming"

    print(f"\nQuery: '{test_query}'")
    print("\nTest Corpus:")
    for i, doc in enumerate(test_corpus):
        print(f"{i+1}. [{len(doc.split())} words] {doc}")

    bm25_test = BM25(k1=1.5, b=0.75)
    bm25_test.fit(test_corpus)
    scores = bm25_test.get_scores(test_query)

    print("\nBM25 Scores (with length normalization, b=0.75):")
    for i, score in enumerate(scores):
        print(f"  Doc {i+1}: {score:.4f}")

    # Without length normalization
    bm25_no_norm = BM25(k1=1.5, b=0.0)
    bm25_no_norm.fit(test_corpus)
    scores_no_norm = bm25_no_norm.get_scores(test_query)

    print("\nBM25 Scores (without length normalization, b=0.0):")
    for i, score in enumerate(scores_no_norm):
        print(f"  Doc {i+1}: {score:.4f}")


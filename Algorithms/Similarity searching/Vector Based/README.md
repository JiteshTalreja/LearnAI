# Vector-Based Similarity Search Algorithms

This folder contains implementations of vector-based (embedding-based) similarity search algorithms. These algorithms convert text into numerical vectors and use mathematical operations to measure similarity.

## Table of Contents
- [TF-IDF (Term Frequency-Inverse Document Frequency)](#tf-idf-term-frequency-inverse-document-frequency)
- [BM25 (Best Matching 25)](#bm25-best-matching-25)
- [SBERT (Sentence-BERT Semantic Embeddings)](#sbert-sentence-bert-semantic-embeddings)
- [Comparison Table](#comparison-of-vector-based-algorithms)

---

## TF-IDF (Term Frequency-Inverse Document Frequency)

### Algorithm Overview

**TF-IDF** is a numerical statistic that reflects how important a word is to a document in a collection (corpus). It's one of the most popular term-weighting schemes in information retrieval.

**Core Concept:**
Words that appear frequently in a document but rarely across the corpus are more important and should have higher weights.

### Mathematical Foundation

**1. Term Frequency (TF):**
```
TF(t, d) = (Number of times term t appears in document d) / (Total terms in d)
```

**Variations:**
- **Raw Count**: Just the frequency
- **Boolean**: 1 if present, 0 otherwise
- **Log Normalization**: 1 + log(frequency)
- **Augmented Frequency**: 0.5 + 0.5 × (frequency / max_frequency)

**2. Inverse Document Frequency (IDF):**
```
IDF(t, D) = log(Total documents / Documents containing term t)
```

The IDF component:
- Increases weight for rare terms
- Decreases weight for common terms (e.g., "the", "is", "and")
- Prevents common words from dominating

**3. Final TF-IDF Score:**
```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)
```

### Algorithm Steps

1. **Tokenization**: Split documents into terms/words
2. **Build Vocabulary**: Create mapping of terms to indices
3. **Compute TF**: Calculate term frequencies for each document
4. **Compute IDF**: Calculate inverse document frequencies across corpus
5. **Multiply**: TF × IDF for each term in each document
6. **Vector Representation**: Each document becomes a vector of TF-IDF scores

### Example

Given corpus:
```
Doc1: "the cat sat on the mat"
Doc2: "the dog sat on the log"
Doc3: "cats and dogs are animals"
```

For term "cat" in Doc1:
```
TF = 1/6 = 0.167 (appears once, document has 6 words)
IDF = log(3/1) = 1.099 (3 total docs, "cat" in 1 doc)
TF-IDF = 0.167 × 1.099 = 0.183
```

For term "the" in Doc1:
```
TF = 2/6 = 0.333 (appears twice)
IDF = log(3/3) = 0 (appears in all documents)
TF-IDF = 0.333 × 0 = 0 (common words get zero weight)
```

### Properties

- **Range**: Typically 0 to positive real numbers
- **Sparsity**: Most values are zero (words don't appear in most documents)
- **High Scores**: Frequent in document, rare in corpus
- **Low Scores**: Common across many documents
- **Time Complexity**: O(n × m) where n = documents, m = vocabulary size
- **Space Complexity**: O(n × m) for full matrix

### Applications

- **Search Engines**: Ranking documents by relevance
- **Document Classification**: Feature extraction for ML models
- **Text Summarization**: Identifying important sentences
- **Keyword Extraction**: Finding representative terms
- **Recommendation Systems**: Content-based filtering
- **Spam Detection**: Identifying spam characteristics

### Code Implementation

**File:** `TFIDF.py`

The implementation provides a complete TF-IDF vectorizer:

#### Core Class: `TFIDFVectorizer`

**Initialization Parameters:**
```python
TFIDFVectorizer(
    use_idf=True,          # Enable IDF weighting
    smooth_idf=True,       # Add 1 to prevent zero division
    sublinear_tf=False,    # Use 1 + log(tf) scaling
    max_features=None,     # Limit vocabulary size
    lowercase=True,        # Convert to lowercase
    stop_words=None        # Words to ignore
)
```

**Key Methods:**

1. **`tokenize(text)`**:
   - Splits text into word tokens
   - Applies lowercasing if enabled
   - Filters stop words
   - Uses regex to extract alphanumeric words

2. **`build_vocabulary(documents)`**:
   - Creates term-to-index mapping
   - Optionally limits to top N features
   - Counts document frequencies for each term

3. **`compute_tf(document)`**:
   - Calculates term frequency vector
   - Supports standard and sublinear TF
   - Normalizes by document length
   - Returns NumPy array

4. **`compute_idf(documents)`**:
   - Calculates IDF for all terms
   - Supports smooth IDF variant
   - Returns NumPy array of IDF values

5. **`fit(documents)`**:
   - Learns vocabulary from corpus
   - Computes IDF values
   - Prepares vectorizer for transformation

6. **`transform(documents)`**:
   - Converts documents to TF-IDF vectors
   - Returns matrix of shape (n_documents, n_features)
   - Uses pre-computed vocabulary and IDF

7. **`fit_transform(documents)`**:
   - Combines fit() and transform()
   - Convenience method for training data

8. **`get_feature_names()`**:
   - Returns list of terms in vocabulary
   - Sorted by index for consistency

#### Helper Functions:

1. **`cosine_similarity(vec1, vec2)`**:
   - Computes similarity between two vectors
   - Returns value between -1 and 1 (typically 0 to 1)
   - Formula: dot(v1, v2) / (norm(v1) × norm(v2))

2. **`compute_similarity_matrix(tfidf_matrix)`**:
   - Computes pairwise similarities for all documents
   - Returns symmetric matrix
   - Efficient batch computation

3. **`get_top_terms(tfidf_vector, feature_names, top_n)`**:
   - Extracts most important terms from a document
   - Returns list of (term, score) tuples
   - Sorted by TF-IDF score

### Key Features

- **Flexible Configuration**: Multiple TF and IDF variants
- **Efficient NumPy Operations**: Fast matrix computations
- **Stop Word Filtering**: Removes common words
- **Vocabulary Control**: Limit features to most important terms
- **Sublinear TF Scaling**: Prevents over-weighting of frequent terms
- **Smooth IDF**: Prevents zero divisions and instability

### Example Usage

```python
from TFIDF import TFIDFVectorizer, cosine_similarity, get_top_terms

# Create vectorizer with stop words
vectorizer = TFIDFVectorizer(
    use_idf=True,
    smooth_idf=True,
    lowercase=True,
    stop_words={'the', 'is', 'on', 'a', 'and'}
)

# Sample documents
documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Machine learning uses Python"
]

# Fit and transform
tfidf_matrix = vectorizer.fit_transform(documents)
print(f"Shape: {tfidf_matrix.shape}")  # (3, vocabulary_size)

# Get top terms for first document
feature_names = vectorizer.get_feature_names()
top_terms = get_top_terms(tfidf_matrix[0], feature_names, top_n=3)
print(f"Top terms: {top_terms}")

# Search with query
query = "cat on mat"
query_vector = vectorizer.transform([query])[0]

# Find most similar document
similarities = []
for i in range(len(documents)):
    sim = cosine_similarity(query_vector, tfidf_matrix[i])
    similarities.append((i, sim))

similarities.sort(key=lambda x: x[1], reverse=True)
print(f"Most similar: Doc {similarities[0][0]}")
```

### Advanced Usage

**Sublinear TF Scaling:**
```python
# Prevents over-weighting of high-frequency terms
vectorizer = TFIDFVectorizer(sublinear_tf=True)
# Uses 1 + log(tf) instead of tf
```

**Limiting Vocabulary:**
```python
# Keep only top 1000 most common terms
vectorizer = TFIDFVectorizer(max_features=1000)
```

**Document Similarity Matrix:**
```python
from TFIDF import compute_similarity_matrix

similarity_matrix = compute_similarity_matrix(tfidf_matrix)
# Returns (n_docs, n_docs) matrix of all pairwise similarities
```

---

## BM25 (Best Matching 25)

### Algorithm Overview

**BM25** is a probabilistic ranking function that improves upon TF-IDF. It's considered state-of-the-art for term-based retrieval and is used by major search engines (including Elasticsearch by default).

**Key Improvements over TF-IDF:**
1. **Term Frequency Saturation**: Uses a saturation function (diminishing returns)
2. **Document Length Normalization**: Fairly penalizes very long documents
3. **Tunable Parameters**: Customizable for different domains

### Mathematical Foundation

**BM25 Formula:**
```
BM25(q, d) = Σ IDF(qi) × [f(qi,d) × (k1 + 1)] / [f(qi,d) + k1 × (1 - b + b × |d|/avgdl)]
```

**Components:**

1. **Query Terms (qi)**: Terms in the search query
2. **Term Frequency f(qi, d)**: Count of term qi in document d
3. **Document Length |d|**: Number of words in document
4. **Average Document Length (avgdl)**: Mean length across corpus

**IDF Component (slightly different from TF-IDF):**
```
IDF(qi) = log[(N - n(qi) + 0.5) / (n(qi) + 0.5) + 1]
```
Where:
- N = total documents
- n(qi) = documents containing qi

### Key Parameters

**1. k1 (Term Frequency Saturation)**
- Controls how quickly score saturates with TF
- Range: 1.2 to 2.0 (typical)
- Default: 1.5
- Higher k1 → TF has more impact
- Lower k1 → faster saturation

**Effect of k1:**
```
k1=0.5:  Aggressive saturation (TF matters less)
k1=1.5:  Balanced (default)
k1=3.0:  Slow saturation (TF matters more)
```

**2. b (Length Normalization)**
- Controls document length penalty
- Range: 0 to 1
- Default: 0.75
- b=0 → no length normalization
- b=1 → full length normalization

**Effect of b:**
```
b=0.0:  Long documents not penalized
b=0.75: Balanced normalization (default)
b=1.0:  Strong penalty for long documents
```

### Term Frequency Saturation Explained

Unlike TF-IDF where TF grows linearly, BM25 uses diminishing returns:

```
Term appears:  1   2   3   4   5   times
TF-IDF score:  1   2   3   4   5   (linear)
BM25 score:    1  1.7 2.1 2.4 2.6  (saturating)
```

This prevents documents with many repetitions from dominating.

### Algorithm Steps

1. **Preprocessing**: Tokenize all documents
2. **Statistics**: Calculate average document length
3. **Document Frequencies**: Count documents containing each term
4. **IDF Calculation**: Compute IDF for each unique term
5. **Query Processing**: Tokenize query
6. **Scoring**: For each document, sum BM25 scores for query terms

### Example

Given corpus and query:

```
Corpus:
Doc1: "machine learning" (2 words)
Doc2: "machine learning is a subset of AI" (7 words)
Doc3: "deep learning machine learning algorithms" (5 words)

Query: "machine learning"
Average doc length: 4.67 words
```

For "machine" in Doc3:
```
f(machine, Doc3) = 2 (appears twice)
|Doc3| = 5 words
k1 = 1.5, b = 0.75

Numerator: 2 × (1.5 + 1) = 5.0
Denominator: 2 + 1.5 × (1 - 0.75 + 0.75 × 5/4.67)
            = 2 + 1.5 × 1.05 = 3.58
Score contribution: IDF × (5.0 / 3.58)
```

### Properties

- **Non-linear**: Saturation prevents over-weighting
- **Length-aware**: Normalizes for document length
- **Probabilistic**: Based on probability ranking principle
- **Tunable**: Parameters optimizable per domain
- **Time Complexity**: O(q × n) where q = query length, n = documents
- **Space Complexity**: O(vocabulary_size)

### Applications

- **Search Engines**: Elasticsearch, Solr (default ranking)
- **Question Answering**: Document retrieval for QA
- **Information Retrieval**: Academic paper search
- **E-commerce**: Product search and ranking
- **Enterprise Search**: Internal document retrieval
- **RAG Systems**: Retrieval for LLMs

### Code Implementation

**File:** `BM25.py`

The implementation provides a complete BM25 ranker:

#### Core Class: `BM25`

**Initialization Parameters:**
```python
BM25(
    k1=1.5,              # TF saturation parameter
    b=0.75,              # Length normalization
    epsilon=0.25,        # IDF floor value
    lowercase=True,      # Case normalization
    stop_words=None      # Words to ignore
)
```

**Key Methods:**

1. **`tokenize(text)`**:
   - Splits text into word tokens
   - Applies case normalization
   - Filters stop words
   - Returns list of tokens

2. **`fit(corpus)`**:
   - Tokenizes all documents
   - Calculates average document length
   - Computes document frequencies
   - Calculates IDF for each term
   - Stores tokenized corpus

3. **`get_scores(query)`**:
   - Tokenizes query
   - Calculates BM25 score for each document
   - Returns NumPy array of scores
   - Implements full BM25 formula

4. **`get_top_n(query, n=5)`**:
   - Returns top N documents for query
   - Returns list of (index, score) tuples
   - Sorted by score descending

5. **`get_batch_scores(queries)`**:
   - Processes multiple queries efficiently
   - Returns matrix of shape (n_queries, n_documents)
   - Useful for batch evaluation

#### Helper Function:

**`compare_bm25_tfidf(corpus, query, k1, b)`**:
- Compares BM25 vs TF-IDF scoring
- Uses same query and corpus
- Returns both score arrays
- Useful for understanding differences

### Key Features

- **Tunable Parameters**: Optimize k1 and b for your domain
- **Efficient Scoring**: Vectorized operations with NumPy
- **Batch Processing**: Handle multiple queries at once
- **IDF Floor**: Epsilon prevents negative IDF values
- **Stop Word Support**: Filter common words
- **Complete Statistics**: Tracks all necessary corpus statistics

### Example Usage

```python
from BM25 import BM25

# Sample corpus
corpus = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Machine learning uses Python",
    "Deep learning is part of machine learning"
]

# Initialize BM25
bm25 = BM25(
    k1=1.5,
    b=0.75,
    stop_words={'the', 'is', 'on', 'a'}
)

# Fit on corpus
bm25.fit(corpus)

# Query
query = "machine learning"
scores = bm25.get_scores(query)
print(f"Scores: {scores}")

# Get top 3 documents
top_docs = bm25.get_top_n(query, n=3)
for rank, (idx, score) in enumerate(top_docs, 1):
    print(f"{rank}. [{score:.4f}] {corpus[idx]}")
```

### Parameter Tuning

**Testing Different k1 Values:**
```python
for k1_val in [0.5, 1.2, 1.5, 2.0, 3.0]:
    bm25 = BM25(k1=k1_val, b=0.75)
    bm25.fit(corpus)
    scores = bm25.get_scores(query)
    print(f"k1={k1_val}: Top score = {max(scores):.4f}")
```

**Testing Different b Values:**
```python
for b_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
    bm25 = BM25(k1=1.5, b=b_val)
    bm25.fit(corpus)
    scores = bm25.get_scores(query)
    print(f"b={b_val}: Top score = {max(scores):.4f}")
```

### BM25 vs TF-IDF Comparison

**Key Differences:**

| Aspect | TF-IDF | BM25 |
|--------|--------|------|
| TF Growth | Linear | Saturating |
| Length Norm | Not standard | Built-in (parameter b) |
| Parameters | Fixed | Tunable (k1, b) |
| IDF Formula | Standard log | Probabilistic variant |
| Domain Adapt | Limited | Excellent (tune params) |

**When to Use BM25:**
- Search applications (better ranking)
- Variable document lengths
- Need parameter tuning
- Probabilistic framework preferred

**When to Use TF-IDF:**
- Document classification (features)
- Simple similarity tasks
- No parameter tuning needed
- Established baseline

### Batch Query Processing

```python
# Process multiple queries efficiently
queries = [
    "machine learning",
    "Python programming",
    "deep neural networks"
]

batch_scores = bm25.get_batch_scores(queries)
# Returns matrix: (3 queries, N documents)

for i, query in enumerate(queries):
    top_idx = batch_scores[i].argmax()
    print(f"Query '{query}' → Doc {top_idx}")
```

### Advanced Features

**Document Length Analysis:**
```python
# See impact of length normalization
print(f"Average doc length: {bm25.avgdl:.2f}")
print(f"Doc lengths: {bm25.doc_len}")

# Compare with/without normalization
bm25_norm = BM25(k1=1.5, b=0.75)
bm25_no_norm = BM25(k1=1.5, b=0.0)

bm25_norm.fit(corpus)
bm25_no_norm.fit(corpus)

scores_norm = bm25_norm.get_scores(query)
scores_no_norm = bm25_no_norm.get_scores(query)
```

---

## SBERT (Sentence-BERT Semantic Embeddings)

### Algorithm Overview

**Sentence-BERT (SBERT)** produces semantically meaningful sentence embeddings by pooling the output of transformer models (BERT, DistilBERT, etc.). Unlike TF-IDF/BM25 which rely on keyword matching, SBERT captures semantic meaning - understanding that "How do I learn coding?" is similar to "Python programming tutorial" even with no word overlap.

**Core Concept:**
Transform entire sentences into dense vectors where cosine similarity correlates with semantic similarity.

### How It Works

```
Input: "Machine learning is fascinating"
         ↓
   [Tokenization]
         ↓
   [CLS] machine learn ##ing is fascin ##ating [SEP]
         ↓
   [Transformer Encoding - BERT/DistilBERT]
         ↓
   Token Embeddings: [e_CLS, e_machine, e_learn, ..., e_SEP]
         ↓
   [Mean Pooling]
         ↓
   Sentence Vector: [0.12, -0.34, 0.56, ...] (768 dimensions)
```

### Mathematical Foundation

**1. Transformer Encoding:**
Each token gets a contextual embedding based on surrounding words:
```
token_embeddings = Transformer(input_tokens)
# Shape: (sequence_length, hidden_dim)
```

**2. Mean Pooling:**
Average all token embeddings (excluding padding):
```
sentence_embedding = Σ(token_embeddings × attention_mask) / Σ(attention_mask)
```

**3. Cosine Similarity:**
Compare two sentence vectors:
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

### Key Properties

| Property | Description |
|----------|-------------|
| **Vector Dimensions** | 768 (BERT/DistilBERT) or 384 (MiniLM) |
| **Semantic Understanding** | Captures meaning beyond exact keywords |
| **Context-Aware** | Same word has different embeddings in different contexts |
| **Pre-trained** | Uses transfer learning from large text corpora |

### Advantages over TF-IDF/BM25

| Aspect | TF-IDF/BM25 | SBERT |
|--------|-------------|-------|
| Matching Type | Keyword-based | Semantic |
| Synonyms | �� Misses them | ✅ Understands them |
| Paraphrases | ❌ Misses them | ✅ Recognizes them |
| Speed | Very fast | Slower (needs GPU for large scale) |
| Interpretability | High | Lower (black box) |

### Code Implementation

**File:** `SBERT.py`

This is a **manual implementation** using `transformers` and `torch` directly (not the sentence-transformers package), showing exactly how SBERT works under the hood.

#### Core Class: `SimpleSBERT`

**Initialization:**
```python
SimpleSBERT(
    model_name="distilbert-base-uncased",  # HuggingFace model
    device=None  # Auto-detect CPU/GPU
)
```

**Key Methods:**

1. **`mean_pooling(token_embeddings, attention_mask)`**:
   - Applies mean pooling over token embeddings
   - Ignores padding tokens using attention mask
   - Returns single vector per sentence

2. **`encode(sentences, batch_size=8, normalize=True)`**:
   - Encodes multiple sentences into embeddings
   - Processes in batches for efficiency
   - Optionally L2-normalizes vectors
   - Returns NumPy array of shape (n_sentences, embedding_dim)

3. **`encode_single(sentence)`**:
   - Convenience method for single sentence
   - Returns 1D NumPy array

#### Helper Functions:

1. **`cosine_similarity(vec1, vec2)`**: Compute similarity between two vectors
2. **`compute_similarity_matrix(embeddings)`**: Pairwise similarity for all documents
3. **`rank_documents(query_embedding, doc_embeddings, top_k)`**: Rank by similarity
4. **`semantic_search(query, documents, embedder, ...)`**: End-to-end search pipeline

### Example Usage

```python
from SBERT import SimpleSBERT, semantic_search, compute_similarity_matrix

# Initialize (downloads model on first run ~250MB)
embedder = SimpleSBERT(model_name="distilbert-base-uncased")

# Sample documents
documents = [
    "Python is a programming language",
    "Machine learning uses Python",
    "Cats and dogs are animals"
]

# Encode documents
doc_embeddings = embedder.encode(documents)
print(f"Shape: {doc_embeddings.shape}")  # (3, 768)

# Compute similarity matrix
sim_matrix = compute_similarity_matrix(doc_embeddings)
print(f"Doc 1 & Doc 2 similarity: {sim_matrix[0, 1]:.4f}")

# Semantic search
query = "How to learn coding?"
results = semantic_search(query, documents, embedder, doc_embeddings, top_k=2)

for idx, score, text in results:
    print(f"[{score:.4f}] {text}")
```

### Understanding Mean Pooling

The key innovation in SBERT is how it creates sentence vectors from token embeddings:

```python
def mean_pooling(token_embeddings, attention_mask):
    # token_embeddings: (batch, seq_len, hidden_dim)
    # attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
    
    # Expand mask to match embedding dimensions
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    
    # Sum embeddings (weighted by mask to ignore padding)
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
    
    # Divide by number of real tokens
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    
    return sum_embeddings / sum_mask
```

### Model Options

| Model | Parameters | Dimensions | Speed | Quality |
|-------|------------|------------|-------|---------|
| `distilbert-base-uncased` | 66M | 768 | Fast | Good |
| `bert-base-uncased` | 110M | 768 | Medium | Better |
| `bert-large-uncased` | 340M | 1024 | Slow | Best |

### Applications

- **Semantic Search**: Find documents by meaning, not just keywords
- **Paraphrase Detection**: Identify sentences with same meaning
- **Question Answering**: Match questions to relevant answers
- **Duplicate Detection**: Find near-duplicate content
- **Document Clustering**: Group by topic/meaning
- **Recommendation Systems**: Content-based recommendations

### Tips for Production

1. **Cache Embeddings**: Pre-compute and store document embeddings
2. **Batch Processing**: Use appropriate batch sizes (8-32)
3. **GPU Acceleration**: Use CUDA for large-scale encoding
4. **Hybrid Search**: Combine with BM25 for best results (keyword + semantic)
5. **Vector Databases**: Use FAISS, Pinecone, or Weaviate for large-scale search

### Dependencies

```bash
pip install torch transformers numpy
```

Note: First run downloads the model (~250MB for DistilBERT).

---

## Comparison of Vector-Based Algorithms

| Algorithm | Type | Saturation | Length Norm | Parameters | Best For |
|-----------|------|------------|-------------|------------|----------|
| **TF-IDF** | Sparse Vector | No | Optional | None | Classification, keyword search |
| **BM25** | Probabilistic | Yes | Yes | k1, b | Search engines, ranking |
| **SBERT** | Dense Embedding | N/A (semantic) | N/A | Model choice | Semantic search, paraphrase mining |

### When to Use Each:

**TF-IDF:**
- ✅ Document classification and clustering
- ✅ Feature extraction for ML models
- ✅ Quick baseline implementation
- ✅ When interpretability is key
- ❌ Not ideal for search ranking

**BM25:**
- ✅ Search and information retrieval
- ✅ Question answering systems
- ✅ When document lengths vary widely
- ✅ When you can tune parameters
- ❌ Overkill for simple classification

**SBERT:**
- ✅ Semantic search and ranking
- ✅ Paraphrase and duplicate detection
- ✅ Sentence clustering and topic modeling
- ✅ When using transformer-based models
- ❌ Requires more resources (GPU recommended)

### Performance Comparison

```
Metric          TF-IDF    BM25      SBERT
Speed           Fast      Fast      Moderate
Memory          O(n×m)    O(n×m)    O(n×d) + model size
Ranking Quality Good      Better    Best
Customization   Low       High      Model/params
Complexity      Simple    Moderate  High
```

---

## Running the Examples

Each file contains comprehensive examples. To run:

```bash
# Run TF-IDF examples
python TFIDF.py

# Run BM25 examples
python BM25.py

# Run SBERT examples
python SBERT.py
```

Expected output includes:
- Vectorization demonstrations
- Top terms extraction
- Document similarity matrices
- Query search examples
- Parameter sensitivity analysis

---

## Dependencies

All implementations require:
- **NumPy**: For efficient vector operations
- **Python 3.7+**: For type hints
- **Standard library**: `typing`, `re`, `collections`

For SBERT (optional):
- **torch**: PyTorch for tensor operations
- **transformers**: HuggingFace transformers for pre-trained models

Install dependencies:
```bash
# Core dependencies (TF-IDF, BM25)
pip install numpy

# Additional for SBERT
pip install torch transformers
```

---

## Integration Examples

### Building a Search Engine

```python
from BM25 import BM25

class SimpleSearchEngine:
    def __init__(self, documents):
        self.documents = documents
        self.bm25 = BM25(k1=1.5, b=0.75)
        self.bm25.fit(documents)
    
    def search(self, query, top_k=5):
        results = self.bm25.get_top_n(query, n=top_k)
        return [(self.documents[idx], score) 
                for idx, score in results]

# Usage
docs = ["doc1 content", "doc2 content", ...]
engine = SimpleSearchEngine(docs)
results = engine.search("my query")
```

### Document Classification Pipeline

```python
from TFIDF import TFIDFVectorizer
from sklearn.linear_model import LogisticRegression

# Extract features
vectorizer = TFIDFVectorizer(max_features=1000)
X = vectorizer.fit_transform(documents)

# Train classifier
clf = LogisticRegression()
clf.fit(X, labels)

# Predict
new_doc_vec = vectorizer.transform([new_document])
prediction = clf.predict(new_doc_vec)
```

---

## Further Reading

### TF-IDF:
- [TF-IDF - Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- "Introduction to Information Retrieval" - Manning, Raghavan, Schütze
- [Scikit-learn TF-IDF Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)

### BM25:
- [BM25 - Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25)
- "The Probabilistic Relevance Framework: BM25 and Beyond" - Robertson & Zaragoza
- [Elasticsearch BM25 Similarity](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-modules-similarity.html)

### General Resources:
- "Information Retrieval: Implementing and Evaluating Search Engines" - Büttcher, Clarke, Cormack
- [Stanford CS276: Information Retrieval](http://web.stanford.edu/class/cs276/)

---

## Best Practices

1. **Preprocessing**: Always lowercase and remove stop words for better results
2. **Parameter Tuning**: For BM25, tune k1 and b on validation set
3. **Vocabulary Control**: Use max_features to limit vocabulary size
4. **Normalization**: Consider L2 normalization for TF-IDF vectors
5. **Evaluation**: Use metrics like NDCG, MAP, MRR for ranking quality

---

## Future Enhancements

These implementations can be extended with:
- **Sparse Matrix Support**: For memory efficiency with large corpora
- **Phrase Queries**: Support for multi-word phrases
- **Field Weighting**: Different weights for title vs. body
- **Query Expansion**: Synonym and related term expansion
- **Learning to Rank**: ML-based re-ranking on top of BM25

---

**Note:** While neural embedding models (BERT, Sentence-BERT) are powerful for semantic search, TF-IDF and BM25 remain essential for:
- Exact keyword matching
- Interpretable ranking
- Low-latency requirements
- Resource-constrained environments
- Hybrid search (combining with embeddings)

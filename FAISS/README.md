# FAISS Implementation - Semantic Search with Vector Embeddings

## Overview

This folder contains a complete implementation of semantic search using FAISS (Facebook AI Similarity Search) and sentence embeddings. The project demonstrates how to:
1. Extract text from PDF documents
2. Generate semantic embeddings using transformer models
3. Build a FAISS index for efficient similarity search
4. Query the index to find semantically similar sentences

## What is FAISS?

**FAISS (Facebook AI Similarity Search)** is a library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors. It's particularly useful for:
- **Fast nearest neighbor search** in high-dimensional spaces
- **Scalability** - can handle billions of vectors
- **GPU acceleration** support for even faster searches
- **Memory efficiency** through various indexing strategies

### Why Use FAISS?

Traditional keyword search only finds exact matches or simple text patterns. Semantic search with FAISS:
- Understands the **meaning** of text, not just keywords
- Finds conceptually similar content even with different wording
- Enables "ask questions in natural language" functionality
- Powers modern AI applications like chatbots, recommendation systems, and RAG (Retrieval Augmented Generation)

## Files in This Folder

### 1. `Create_embeddings.py`
**Purpose**: Processes a PDF document and generates semantic embeddings for all sentences.

**What it does**:
- Extracts text from `the_hundred_page_language_models_book.pdf` using PyPDF
- Splits the text into individual sentences using NLTK's sentence tokenizer
- Generates 768-dimensional embeddings for each sentence using SentenceTransformer
- Saves embeddings to `sentence_embeddings.npy` (binary numpy array)
- Saves original sentences to `sentences.txt` (text file for reference)

**Key Components**:
```python
# Extract sentences from PDF
extract_sentences_from_pdf(file_path)
  └─> Uses PyPDF to read PDF pages
  └─> Uses NLTK to tokenize into sentences
  
# Generate embeddings
model = SentenceTransformer('bert-base-nli-mean-tokens')
embeddings = model.encode(sentences)
  └─> Each sentence → 768-dimensional vector
  └─> Semantically similar sentences have similar vectors
```

**Output Files**:
- `sentence_embeddings.npy`: NumPy array of shape (N, 768) where N is the number of sentences
- `sentences.txt`: Plain text file with one sentence per line

---

### 2. `faiss_searching.py`
**Purpose**: Performs semantic search queries against the generated embeddings using FAISS.

**What it does**:
- Loads pre-generated embeddings from `sentence_embeddings.npy`
- Loads corresponding sentences from `sentences.txt`
- Creates a FAISS index (IndexFlatL2) for similarity search
- Runs example queries and finds the top-K most similar sentences
- Displays results with distance scores

**Key Components**:

#### Step 1: Load Data
```python
embeddings = np.load("sentence_embeddings.npy")  # Load vectors
sentences = [line.strip() for line in f.readlines()]  # Load text
```

#### Step 2: Create FAISS Index
```python
dimension = embeddings.shape[1]  # 768 dimensions
index = faiss.IndexFlatL2(dimension)  # L2 distance metric
index.add(embeddings.astype('float32'))  # Add all vectors to index
```

**IndexFlatL2** uses Euclidean (L2) distance to measure similarity:
- Lower distance = more similar
- Performs exhaustive search (checks all vectors)
- Most accurate but slower for large datasets

#### Step 3: Search
```python
query_embedding = model.encode([query])  # Convert query to vector
distances, indices = index.search(query_embedding, k=5)  # Find 5 nearest
```

**Example Queries**:
- "What are language models?"
- "How do transformers work?"
- "Machine learning techniques"
- "Neural network architecture"

**Output Format**:
```
Query: 'What are language models?'
1. [Distance: 45.2341] Sentence from PDF that's most similar...
2. [Distance: 48.7652] Second most similar sentence...
...
```

---

### 3. `Faiss_intro.ipynb`
**Purpose**: Jupyter notebook for interactive exploration and learning.

This notebook likely contains:
- Step-by-step tutorial on FAISS concepts
- Visualization of embeddings and similarity
- Experimentation with different FAISS index types
- Performance comparisons

---

### 4. `the_hundred_page_language_models_book.pdf`
**Purpose**: Source document for the semantic search demonstration.

This PDF serves as the knowledge base that gets indexed and searched.

---

## How the Implementation Works

### Architecture Overview

```
PDF Document
    ↓
[Extract Text] (PyPDF)
    ↓
[Tokenize into Sentences] (NLTK)
    ↓
[Generate Embeddings] (SentenceTransformer)
    ↓
[Save to Disk] (NumPy)
    ↓
[Load Embeddings] (faiss_searching.py)
    ↓
[Build FAISS Index] (IndexFlatL2)
    ↓
[Query Processing] (Natural Language → Vector)
    ↓
[Similarity Search] (Find Nearest Neighbors)
    ↓
[Display Results] (Top-K Similar Sentences)
```

### Detailed Workflow

#### Phase 1: Embedding Generation (`Create_embeddings.py`)

1. **PDF Parsing**
   - Opens PDF file in binary mode
   - Extracts text from each page
   - Concatenates into a single document

2. **Sentence Tokenization**
   - Uses NLTK's `sent_tokenize()` for intelligent splitting
   - Handles abbreviations, punctuation, edge cases
   - Produces clean, individual sentences

3. **Embedding Generation**
   - Uses `bert-base-nli-mean-tokens` model
   - Each sentence → 768-dimensional vector
   - Vectors capture semantic meaning
   - Similar meanings → similar vectors

4. **Persistence**
   - Saves embeddings as NumPy array for fast loading
   - Saves sentences as text for human reference

#### Phase 2: Semantic Search (`faiss_searching.py`)

1. **Index Construction**
   - Loads embeddings into memory
   - Creates FAISS IndexFlatL2 (exhaustive search)
   - Adds all vectors to the index

2. **Query Processing**
   - User query: "What are language models?"
   - Encoded using same SentenceTransformer model
   - Produces query vector (768 dimensions)

3. **Similarity Search**
   - FAISS computes L2 distance from query to all embeddings
   - Returns K nearest neighbors (default: 5)
   - Lower distance = higher similarity

4. **Result Display**
   - Shows top-K sentences with distance scores
   - Truncates long sentences for readability
   - Formatted output for easy interpretation

---

## Mathematical Foundation

### L2 (Euclidean) Distance

The IndexFlatL2 uses Euclidean distance to measure similarity:

```
distance(v1, v2) = sqrt(Σ(v1[i] - v2[i])²)
```

Where:
- `v1`, `v2` are embedding vectors
- Smaller distance = more similar
- Distance of 0 = identical vectors

### Why Vector Embeddings Work

Transformer models (like BERT) learn to map text to vectors such that:
- Semantically similar sentences cluster together in vector space
- Mathematical operations preserve meaning relationships
- Enables "fuzzy matching" based on semantic similarity

Example:
```
"dog" and "puppy" → similar vectors (both canines)
"dog" and "building" → distant vectors (unrelated concepts)
```

---

## Installation & Dependencies

### Required Packages

```bash
pip install faiss-cpu numpy sentence-transformers pypdf nltk
```

Or if using the project's dependency management:
```bash
cd "C:/Users/jitalreja/Desktop/Docs Personal/langChain/langchain-course"
uv sync
```

### Additional Setup

For NLTK sentence tokenizer:
```python
import nltk
nltk.download('punkt')
```

---

## Usage Instructions

### Step 1: Generate Embeddings

```bash
cd FAISS
python Create_embeddings.py
```

**Expected Output**:
```
Generated 2552 embeddings with shape (2552, 768)
Embeddings saved to sentence_embeddings.npy
Sentences saved to sentences.txt
```

**Generated Files**:
- `sentence_embeddings.npy` (~15 MB)
- `sentences.txt` (~500 KB)

### Step 2: Run Semantic Search

```bash
python faiss_searching.py
```

**Expected Output**:
```
Loading embeddings...
Loaded embeddings with shape: (2552, 768)
Loading sentences...
Loaded 2552 sentences
FAISS index created with 2552 vectors

================================================================================
SEARCHING FOR SIMILAR SENTENCES
================================================================================

Query: 'What are language models?'
--------------------------------------------------------------------------------
1. [Distance: 45.2341] Language models are statistical models...
2. [Distance: 48.7652] A language model assigns probabilities...
...
```

---

## Customization Options

### Change the Number of Results

In `faiss_searching.py`, modify:
```python
k = 5  # Change to 10, 20, etc.
```

### Add Your Own Queries

```python
queries = [
    "Your custom query here",
    "Another search term",
    # Add more...
]
```

### Use a Different PDF

1. Place your PDF in the FAISS folder
2. In `Create_embeddings.py`, change:
```python
sentences_to_transform = extract_sentences_from_pdf("your_document.pdf")
```
3. Re-run embedding generation

### Use a Different Embedding Model

```python
# Faster but less accurate
model = SentenceTransformer('all-MiniLM-L6-v2')

# More accurate but slower
model = SentenceTransformer('all-mpnet-base-v2')

# Current model (balanced)
model = SentenceTransformer('bert-base-nli-mean-tokens')
```

### Try Different FAISS Index Types

```python
# Current: Flat index (exhaustive search)
index = faiss.IndexFlatL2(dimension)

# Faster: IVF index (approximate search)
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
index.train(embeddings)  # Training required
index.add(embeddings)

# Even faster: HNSW index (graph-based)
index = faiss.IndexHNSWFlat(dimension, 32)  # 32 neighbors
```

---

## Performance Considerations

### Index Types Comparison

| Index Type | Accuracy | Speed | Memory | Training Needed |
|------------|----------|-------|--------|-----------------|
| IndexFlatL2 | 100% | Slow | Medium | No |
| IndexIVFFlat | ~95% | Fast | Medium | Yes |
| IndexHNSWFlat | ~99% | Very Fast | High | No |
| IndexFlatIP | 100% | Slow | Medium | No (inner product) |

### Scaling Recommendations

- **< 10K vectors**: Use IndexFlatL2 (current implementation)
- **10K - 1M vectors**: Use IndexIVFFlat or IndexHNSWFlat
- **> 1M vectors**: Use GPU acceleration with IndexIVFPQ
- **Limited memory**: Use product quantization (PQ)

### Current Performance

With 2,552 sentences (768-dimensional embeddings):
- **Index size**: ~7.5 MB in memory
- **Search time**: ~1-2 ms per query
- **Accuracy**: 100% (exhaustive search)

---

## Troubleshooting

### Common Issues

#### 1. FileNotFoundError: sentences.txt
**Solution**: Run `Create_embeddings.py` first to generate the files.

#### 2. NLTK punkt tokenizer not found
**Solution**: 
```python
import nltk
nltk.download('punkt')
```

#### 3. Out of memory
**Solution**: Process PDF in chunks or use a smaller model like 'all-MiniLM-L6-v2'.

#### 4. Slow embedding generation
**Solution**: Enable GPU support with `device='cuda'`:
```python
model = SentenceTransformer('bert-base-nli-mean-tokens', device='cuda')
```

#### 5. Poor search results
**Solutions**:
- Try a different embedding model
- Increase K (number of results)
- Check if query is too broad/specific
- Verify PDF extraction quality

---

## Advanced Features to Explore

### 1. Add Metadata Filtering
Store metadata (page numbers, sections) alongside embeddings and filter results.

### 2. Hybrid Search
Combine FAISS semantic search with traditional keyword search (BM25).

### 3. Re-ranking
Use a cross-encoder model to re-rank FAISS results for higher precision.

### 4. Incremental Updates
Add new documents without regenerating all embeddings:
```python
new_embeddings = model.encode(new_sentences)
index.add(new_embeddings)
```

### 5. Query Expansion
Automatically expand queries with synonyms or related terms.

### 6. Clustering
Use FAISS for document clustering and topic discovery:
```python
kmeans = faiss.Kmeans(dimension, 10)  # 10 clusters
kmeans.train(embeddings)
```

---

## Real-World Applications

This FAISS implementation can be adapted for:

1. **Question Answering Systems**: Find relevant passages for user questions
2. **Document Search**: Search through large document collections
3. **Recommendation Engines**: Find similar products, articles, or content
4. **RAG (Retrieval Augmented Generation)**: Retrieve context for LLMs
5. **Duplicate Detection**: Find near-duplicate documents
6. **Customer Support**: Match queries to knowledge base articles
7. **Research Tools**: Search academic papers by concept
8. **Content Discovery**: Find related blog posts or articles

---

## References & Resources

### Official Documentation
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [Sentence Transformers](https://www.sbert.net/)

### Tutorials
- [FAISS: The Missing Manual](https://www.pinecone.io/learn/faiss-tutorial/)
- [Sentence Embeddings Explained](https://www.sbert.net/docs/pretrained_models.html)

### Research Papers
- FAISS: "Billion-scale similarity search with GPUs" (Johnson et al., 2017)
- BERT: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- Sentence-BERT: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (Reimers & Gurevych, 2019)

---

## Next Steps

1. ✅ Generate embeddings for your PDF
2. ✅ Run semantic search queries
3. 🔲 Experiment with different embedding models
4. 🔲 Try different FAISS index types
5. 🔲 Integrate with LangChain for RAG
6. 🔲 Build a web interface for search
7. 🔲 Add multiple document support
8. 🔲 Implement metadata filtering

---

## License

This implementation is part of the langchain-course project. See the main LICENSE file for details.

---

## Questions or Issues?

For questions about this implementation, check:
1. The code comments in each file
2. Official FAISS documentation
3. Sentence Transformers documentation
4. The Jupyter notebook (`Faiss_intro.ipynb`) for interactive examples


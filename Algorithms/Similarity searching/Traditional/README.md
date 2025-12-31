# Traditional Similarity Search Algorithms

This folder contains implementations of traditional (non-vector-based) similarity search algorithms. These algorithms work by directly comparing text or sets without creating dense vector representations.

## Table of Contents
- [Levenshtein Distance](#levenshtein-distance)
- [Jaccard Similarity](#jaccard-similarity)
- [W-Shingling](#w-shingling)

---

## Levenshtein Distance

### Algorithm Overview

The **Levenshtein distance** (also known as edit distance) measures the similarity between two strings by calculating the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one word into another.

**Mathematical Definition:**
```
lev(a, b) = minimum number of operations to transform string a into string b
```

**Algorithm Steps:**
1. **Initialization**: Create a matrix D of size (m+1) × (n+1) where m and n are the lengths of the two strings
2. **Base Cases**: 
   - First row: D[i][0] = i (cost of i deletions)
   - First column: D[0][j] = j (cost of j insertions)
3. **Matrix Filling**: For each cell D[i][j]:
   - If characters match: D[i][j] = D[i-1][j-1] (no operation needed)
   - If characters differ: D[i][j] = 1 + min(
     - D[i-1][j] (deletion from s1)
     - D[i][j-1] (insertion to s1)
     - D[i-1][j-1] (substitution)
   )
4. **Result**: The value at D[m][n] is the Levenshtein distance

**Example:**
```
Transform "kitten" → "sitting"
Operations: 
1. kitten → sitten (substitute 'k' with 's')
2. sitten → sittin (substitute 'e' with 'i')
3. sittin → sitting (insert 'g')
Distance = 3
```

**Properties:**
- Range: 0 to max(len(s1), len(s2))
- Symmetry: distance(s1, s2) = distance(s2, s1)
- Triangle inequality holds
- Time complexity: O(m × n)
- Space complexity: O(m × n) (can be optimized to O(min(m, n)))

**Applications:**
- Spell checking and auto-correction
- DNA sequence analysis
- Plagiarism detection
- Fuzzy string matching
- Data deduplication

### Code Implementation

**File:** `Levenshtein.py`

The implementation provides several key functions:

1. **`levenshtein_distance_numpy(s1, s2)`**: 
   - Core implementation using NumPy for efficient matrix operations
   - Returns the minimum edit distance as an integer

2. **`levenshtein_distance_normalized(s1, s2)`**: 
   - Returns normalized distance (0 to 1) by dividing by max string length
   - Useful for comparing strings of different lengths

3. **`levenshtein_similarity(s1, s2)`**: 
   - Converts distance to similarity score (1 - normalized_distance)
   - Higher values indicate more similar strings

4. **`get_edit_operations(s1, s2)`**: 
   - Traces back through the matrix to find actual edit operations
   - Returns list of operations performed

**Key Features:**
- Uses NumPy for efficient matrix operations
- Provides both distance and similarity metrics
- Includes normalized versions for fair comparison
- Type hints for better code clarity

**Example Usage:**
```python
from Levenshtein import levenshtein_distance_numpy, levenshtein_similarity

# Calculate distance
distance = levenshtein_distance_numpy("kitten", "sitting")
print(f"Distance: {distance}")  # Output: 3

# Calculate similarity
similarity = levenshtein_similarity("kitten", "sitting")
print(f"Similarity: {similarity:.2f}")  # Output: 0.57
```

---

## Jaccard Similarity

### Algorithm Overview

The **Jaccard similarity coefficient** (Jaccard index) measures similarity between finite sets by comparing the size of their intersection to the size of their union.

**Mathematical Definition:**
```
J(A, B) = |A ∩ B| / |A ∪ B|
        = |A ∩ B| / (|A| + |B| - |A ∩ B|)
```

Where:
- A and B are two sets
- |A ∩ B| is the intersection (common elements)
- |A ∪ B| is the union (all unique elements)

**Algorithm Steps:**
1. **Set Creation**: Convert inputs into sets (characters, words, or tokens)
2. **Intersection**: Find elements common to both sets
3. **Union**: Find all unique elements across both sets
4. **Division**: Calculate |intersection| / |union|

**Example:**
```
Compare "hello" and "halo"
Set A: {h, e, l, o} (unique characters)
Set B: {h, a, l, o}
Intersection: {h, l, o} = 3 elements
Union: {h, e, l, o, a} = 5 elements
Jaccard Similarity: 3/5 = 0.6
```

**Jaccard Distance:**
```
J_distance(A, B) = 1 - J(A, B)
```

**Properties:**
- Range: 0 (no overlap) to 1 (identical sets)
- Symmetry: J(A, B) = J(B, A)
- Empty sets: Defined as similarity = 1
- Time complexity: O(|A| + |B|)
- Space complexity: O(|A| + |B|)

**Applications:**
- Text similarity and document comparison
- Recommendation systems
- Data deduplication
- DNA sequence analysis
- Image similarity comparison

### Code Implementation

**File:** `Jaccard.py`

The implementation provides multiple variations:

1. **`jaccard_similarity(s1, s2)`**: 
   - Character-based Jaccard similarity
   - Converts strings to sets of characters
   - Returns similarity coefficient (0 to 1)

2. **`jaccard_distance(s1, s2)`**: 
   - Returns Jaccard distance (1 - similarity)
   - Useful when distance metric is preferred

3. **`jaccard_similarity_tokens(s1, s2)`**: 
   - Word-based (token-based) Jaccard similarity
   - Splits text into words and compares word sets
   - Better for sentence/document comparison
   - Case-insensitive by default

4. **`jaccard_similarity_numpy(s1, s2)`**: 
   - NumPy-based implementation
   - Uses np.intersect1d and np.union1d
   - Demonstrates alternative implementation approach

**Key Features:**
- Multiple granularity levels (character vs. word)
- Both similarity and distance metrics
- NumPy implementation for consistency
- Handles edge cases (empty strings, identical strings)

**Example Usage:**
```python
from Jaccard import jaccard_similarity, jaccard_similarity_tokens

# Character-based comparison
char_sim = jaccard_similarity("hello", "halo")
print(f"Character similarity: {char_sim:.2f}")  # Output: 0.60

# Word-based comparison for sentences
sent1 = "the quick brown fox"
sent2 = "the quick brown dog"
word_sim = jaccard_similarity_tokens(sent1, sent2)
print(f"Word similarity: {word_sim:.2f}")  # Output: 0.75
```

---

## W-Shingling

### Algorithm Overview

**W-Shingling** (k-shingling) is a technique for converting documents into sets for comparison. A shingle is a contiguous subsequence of w tokens (characters or words) from a document.

**Key Concept:**
Instead of treating documents as bags of words, shingling preserves local context by grouping adjacent elements.

**Algorithm Steps:**
1. **Choose Shingle Size (w)**: 
   - Character shingles: typically w = 2-5
   - Word shingles: typically w = 1-3
2. **Extract Shingles**: Slide a window of size w across the document
3. **Create Set**: Store unique shingles (duplicates removed)
4. **Compare**: Use Jaccard similarity on shingle sets

**Example 1: Character 3-shingling**
```
Text: "hello"
Shingles: {"hel", "ell", "llo"}
```

**Example 2: Word 2-shingling**
```
Text: "the cat sat on the mat"
Shingles: {"the cat", "cat sat", "sat on", "on the", "the mat"}
```

**Why W-Shingling?**
- **Preserves Order**: Unlike bag-of-words, maintains sequence information
- **Robust**: Small changes result in only few different shingles
- **Near-Duplicate Detection**: Highly effective for finding similar documents
- **Flexible**: Works at character or word level

**Similarity Calculation:**
```
J(D1, D2) = |Shingles(D1) ∩ Shingles(D2)| / |Shingles(D1) ∪ Shingles(D2)|
```

**Properties:**
- Time complexity: O(n) for shingle extraction, where n is document length
- Space complexity: O(number of unique shingles)
- Increasing w makes algorithm more specific (fewer false positives)
- Decreasing w makes algorithm more sensitive (catches more similarities)

**Applications:**
- Plagiarism detection
- Near-duplicate document detection (web crawling)
- Document clustering
- Copyright infringement detection
- Spam detection

### Code Implementation

**File:** `WShingling.py`

The implementation provides comprehensive shingling functionality:

1. **`character_shingling(text, w)`**: 
   - Extracts character-based shingles of size w
   - Returns set of character n-grams
   - Good for typo detection and near-duplicates

2. **`word_shingling(text, w)`**: 
   - Extracts word-based shingles of size w
   - Tokenizes text into words first
   - Better for semantic similarity
   - Case-insensitive by default

3. **`jaccard_similarity_shingles(shingles1, shingles2)`**: 
   - Calculates Jaccard similarity between two shingle sets
   - Returns coefficient between 0 and 1

4. **`document_similarity_character(doc1, doc2, w=3)`**: 
   - Complete pipeline for character-based comparison
   - Returns (similarity, shingles1, shingles2)
   - Default w=3 is good for most text

5. **`document_similarity_word(doc1, doc2, w=2)`**: 
   - Complete pipeline for word-based comparison
   - Returns (similarity, shingles1, shingles2)
   - Default w=2 captures bigrams

6. **`shingle_vector_representation(shingles, universe)`**: 
   - Converts shingle sets to binary vectors
   - Enables vector-based operations
   - Universe is the set of all possible shingles

7. **`cosine_similarity_shingles(shingles1, shingles2)`**: 
   - Alternative similarity using cosine measure
   - Converts to vectors first, then calculates cosine

**Key Features:**
- Both character and word-level shingling
- Multiple similarity metrics (Jaccard, Cosine)
- Vector representation for advanced operations
- Complete pipeline functions for ease of use
- Handles edge cases properly

**Example Usage:**
```python
from WShingling import document_similarity_word, character_shingling

# Word-based document comparison
doc1 = "the cat sat on the mat"
doc2 = "the cat sat on the floor"
similarity, shingles1, shingles2 = document_similarity_word(doc1, doc2, w=2)
print(f"Similarity: {similarity:.2f}")  # Output: 0.67

# Character-based for near-duplicate detection
shingles = character_shingling("hello world", w=3)
print(f"Character 3-shingles: {shingles}")
# Output: {'hel', 'ell', 'llo', 'lo ', 'o w', ' wo', 'wor', 'orl', 'rld'}

# Plagiarism detection example
original = "to be or not to be that is the question"
paraphrase = "to be or not to be that is the inquiry"
different = "the quick brown fox jumps over lazy dog"

sim1, _, _ = document_similarity_word(original, paraphrase, w=3)
sim2, _, _ = document_similarity_word(original, different, w=3)
print(f"Original vs Paraphrase: {sim1:.2f}")  # High similarity
print(f"Original vs Different: {sim2:.2f}")   # Low similarity
```

---

## Comparison of Traditional Algorithms

| Algorithm | Type | Best For | Time Complexity | Key Advantage |
|-----------|------|----------|----------------|---------------|
| **Levenshtein** | Edit Distance | Typos, spelling | O(m×n) | Exact operations count |
| **Jaccard** | Set-based | Quick comparison | O(|A|+|B|) | Very fast, simple |
| **W-Shingling** | Sequence-based | Near-duplicates | O(n) | Preserves word order |

**When to Use:**
- **Levenshtein**: Use for spell checking, short strings, when you need exact edit count
- **Jaccard**: Use for quick document comparison, when word order doesn't matter
- **W-Shingling**: Use for plagiarism detection, near-duplicate detection, when context matters

---

## Running the Examples

Each file contains example usage at the bottom. To run:

```bash
# Run Levenshtein examples
python Levenshtein.py

# Run Jaccard examples
python Jaccard.py

# Run W-Shingling examples
python WShingling.py
```

## Dependencies

All implementations use:
- **NumPy**: For efficient array operations
- **Python 3.7+**: For type hints and modern features
- **Standard library**: `typing`, `re`, `collections`

Install dependencies:
```bash
pip install numpy
```

---

## Further Reading

- [Levenshtein Distance - Wikipedia](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Jaccard Index - Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)
- [W-Shingling - Stanford CS246](http://infolab.stanford.edu/~ullman/mmds/ch3.pdf)
- "Mining of Massive Datasets" by Leskovec, Rajaraman, and Ullman

---

**Note:** These algorithms form the foundation of many modern information retrieval systems. While newer deep learning approaches exist, these traditional methods remain valuable for their interpretability, efficiency, and reliability.


""""
W-Shingling Algorithm for Document Similarity using NumPy and Sets.

W-Shingling (also known as k-shingling) is a technique used to convert documents into sets for comparison.
A shingle (or w-shingle) is any contiguous subsequence of w tokens (characters or words) in a document.

The algorithm works as follows:
1. Choose a shingle size w (typically 2-5 for characters, 1-3 for words)
2. Extract all contiguous subsequences of length w from the document
3. Create a set of these shingles (removing duplicates)
4. Compare documents by comparing their shingle sets using similarity measures like Jaccard

Why W-Shingling?
- Robust to small variations: Minor changes in a document result in only a few different shingles
- Preserves word order: Unlike bag-of-words, shingling maintains sequence information
- Effective for near-duplicate detection: Similar documents have highly overlapping shingle sets
- Flexible granularity: Can work at character or word level

Applications:
- Plagiarism detection
- Near-duplicate document detection (e.g., web crawling)
- Document clustering
- Copyright infringement detection
- Spam detection

Example: Character 3-shingling of "hello"
Shingles: {"hel", "ell", "llo"}

Example: Word 2-shingling of "the cat sat on the mat"
Shingles: {"the cat", "cat sat", "sat on", "on the", "the mat"}

The similarity between two documents can then be computed using Jaccard similarity:
J(D1, D2) = |Shingles(D1) ∩ Shingles(D2)| / |Shingles(D1) ∪ Shingles(D2)|

"""

import numpy as np
from typing import Set, List, Tuple


def character_shingling(text: str, w: int) -> Set[str]:
    """
    Generates character-based shingles of size w from the input text.

    Args:
        text: The input text string.
        w: The shingle size (number of characters).

    Returns:
        A set of character shingles.
    """
    # Handle edge cases
    if w <= 0 or len(text) < w:
        return set()

    # Generate all character w-shingles
    shingles = set()
    for i in range(len(text) - w + 1):
        shingle = text[i:i + w]
        shingles.add(shingle)

    return shingles


def word_shingling(text: str, w: int) -> Set[str]:
    """
    Generates word-based shingles of size w from the input text.

    Args:
        text: The input text string.
        w: The shingle size (number of words).

    Returns:
        A set of word shingles.
    """
    # Tokenize the text into words
    words = text.lower().split()

    # Handle edge cases
    if w <= 0 or len(words) < w:
        return set()

    # Generate all word w-shingles
    shingles = set()
    for i in range(len(words) - w + 1):
        shingle = " ".join(words[i:i + w])
        shingles.add(shingle)

    return shingles


def jaccard_similarity_shingles(shingles1: Set[str], shingles2: Set[str]) -> float:
    """
    Calculates the Jaccard similarity between two sets of shingles.

    Args:
        shingles1: The first set of shingles.
        shingles2: The second set of shingles.

    Returns:
        The Jaccard similarity coefficient (between 0 and 1).
    """
    # Handle empty sets
    if len(shingles1) == 0 and len(shingles2) == 0:
        return 1.0

    # Calculate intersection and union
    intersection = shingles1.intersection(shingles2)
    union = shingles1.union(shingles2)

    # Calculate Jaccard similarity
    similarity = len(intersection) / len(union) if len(union) > 0 else 0.0

    return similarity


def document_similarity_character(doc1: str, doc2: str, w: int = 3) -> Tuple[float, Set[str], Set[str]]:
    """
    Calculates similarity between two documents using character-based w-shingling.

    Args:
        doc1: The first document.
        doc2: The second document.
        w: The shingle size (default: 3).

    Returns:
        A tuple containing:
        - Jaccard similarity score
        - Set of shingles from doc1
        - Set of shingles from doc2
    """
    shingles1 = character_shingling(doc1, w)
    shingles2 = character_shingling(doc2, w)
    similarity = jaccard_similarity_shingles(shingles1, shingles2)

    return similarity, shingles1, shingles2


def document_similarity_word(doc1: str, doc2: str, w: int = 2) -> Tuple[float, Set[str], Set[str]]:
    """
    Calculates similarity between two documents using word-based w-shingling.

    Args:
        doc1: The first document.
        doc2: The second document.
        w: The shingle size (default: 2).

    Returns:
        A tuple containing:
        - Jaccard similarity score
        - Set of shingles from doc1
        - Set of shingles from doc2
    """
    shingles1 = word_shingling(doc1, w)
    shingles2 = word_shingling(doc2, w)
    similarity = jaccard_similarity_shingles(shingles1, shingles2)

    return similarity, shingles1, shingles2


def shingle_vector_representation(shingles: Set[str], universe: Set[str]) -> np.ndarray:
    """
    Converts a set of shingles into a binary vector representation using NumPy.
    Each dimension corresponds to a shingle in the universe of all possible shingles.

    Args:
        shingles: The set of shingles from a document.
        universe: The universe of all possible shingles (union of all document shingles).

    Returns:
        A binary NumPy array where 1 indicates presence of a shingle, 0 indicates absence.
    """
    # Create a sorted list from the universe for consistent indexing
    universe_list = sorted(list(universe))

    # Create a binary vector
    vector = np.zeros(len(universe_list), dtype=int)

    # Set 1 for shingles that are present
    for i, shingle in enumerate(universe_list):
        if shingle in shingles:
            vector[i] = 1

    return vector


def cosine_similarity_shingles(shingles1: Set[str], shingles2: Set[str]) -> float:
    """
    Calculates cosine similarity between two sets of shingles using vector representations.

    Args:
        shingles1: The first set of shingles.
        shingles2: The second set of shingles.

    Returns:
        The cosine similarity (between 0 and 1).
    """
    # Create universe of all shingles
    universe = shingles1.union(shingles2)

    if len(universe) == 0:
        return 1.0

    # Convert to vectors
    vec1 = shingle_vector_representation(shingles1, universe)
    vec2 = shingle_vector_representation(shingles2, universe)

    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# Example Usage:
print("=" * 70)
print("CHARACTER-BASED W-SHINGLING EXAMPLES")
print("=" * 70)

text1 = "the quick brown fox"
text2 = "the quick brown dog"

print(f"\nDocument 1: '{text1}'")
print(f"Document 2: '{text2}'")

# Character 3-shingling
similarity_char, shingles1_char, shingles2_char = document_similarity_character(text1, text2, w=3)
print(f"\n3-Character Shingles:")
print(f"Doc1 shingles (sample): {list(shingles1_char)[:5]}...")
print(f"Doc2 shingles (sample): {list(shingles2_char)[:5]}...")
print(f"Jaccard Similarity: {similarity_char:.4f}")

print("\n" + "=" * 70)
print("WORD-BASED W-SHINGLING EXAMPLES")
print("=" * 70)

doc1 = "the cat sat on the mat"
doc2 = "the cat sat on the floor"

print(f"\nDocument 1: '{doc1}'")
print(f"Document 2: '{doc2}'")

# Word 2-shingling
similarity_word, shingles1_word, shingles2_word = document_similarity_word(doc1, doc2, w=2)
print(f"\n2-Word Shingles:")
print(f"Doc1 shingles: {sorted(shingles1_word)}")
print(f"Doc2 shingles: {sorted(shingles2_word)}")
print(f"Jaccard Similarity: {similarity_word:.4f}")

# Cosine similarity comparison
cosine_sim = cosine_similarity_shingles(shingles1_word, shingles2_word)
print(f"Cosine Similarity: {cosine_sim:.4f}")

print("\n" + "=" * 70)
print("PLAGIARISM DETECTION EXAMPLE")
print("=" * 70)

original = "to be or not to be that is the question"
paraphrase = "to be or not to be that is the inquiry"
different = "the quick brown fox jumps over the lazy dog"

print(f"\nOriginal: '{original}'")
print(f"Paraphrase: '{paraphrase}'")
print(f"Different: '{different}'")

# Compare using word 3-shingling
sim_para, _, _ = document_similarity_word(original, paraphrase, w=3)
sim_diff, _, _ = document_similarity_word(original, different, w=3)

print(f"\nSimilarity (Original vs Paraphrase): {sim_para:.4f}")
print(f"Similarity (Original vs Different): {sim_diff:.4f}")
print(f"\nConclusion: The paraphrase has {sim_para/sim_diff:.2f}x higher similarity to the original.")


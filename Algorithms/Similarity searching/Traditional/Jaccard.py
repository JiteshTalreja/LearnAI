"""
Jaccard Similarity Calculation using NumPy and Sets.

The Jaccard similarity coefficient, also known as the Jaccard index, is a statistic used for gauging the similarity and diversity of sample sets.
It measures similarity between finite sample sets and is defined as the size of the intersection divided by the size of the union of the sample sets.

The Jaccard similarity coefficient is calculated as:
J(A, B) = |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)

Where:
- A and B are two sets
- |A ∩ B| is the size of the intersection (elements common to both sets)
- |A ∪ B| is the size of the union (all unique elements in both sets)

The Jaccard distance is then calculated as:
J_distance(A, B) = 1 - J(A, B)

Properties:
1. Range: The Jaccard similarity ranges from 0 (no overlap) to 1 (identical sets)
2. Symmetry: J(A, B) = J(B, A)
3. Special Cases:
   - If both sets are empty, the similarity is defined as 1 (identical)
   - If one set is empty and the other is not, the similarity is 0

Applications:
- Text similarity and document comparison
- Recommendation systems
- Data deduplication
- DNA sequence analysis
- Image similarity

Example: "hello" and "halo"
Set A (characters in "hello"): {h, e, l, o}
Set B (characters in "halo"): {h, a, l, o}
Intersection: {h, l, o} = 3 elements
Union: {h, e, l, o, a} = 5 elements
Jaccard Similarity: 3/5 = 0.6

"""

import numpy as np


def jaccard_similarity(s1: str, s2: str) -> float:
    """
    Calculates the Jaccard similarity coefficient between two strings based on character sets.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        The Jaccard similarity coefficient (between 0 and 1).
    """
    # Convert strings to sets of characters
    set1 = set(s1)
    set2 = set(s2)

    # Handle the case where both sets are empty
    if len(set1) == 0 and len(set2) == 0:
        return 1.0

    # Calculate intersection and union
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # Calculate Jaccard similarity
    similarity = len(intersection) / len(union) if len(union) > 0 else 0.0

    return similarity


def jaccard_distance(s1: str, s2: str) -> float:
    """
    Calculates the Jaccard distance between two strings.
    Jaccard distance = 1 - Jaccard similarity

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        The Jaccard distance (between 0 and 1).
    """
    return 1.0 - jaccard_similarity(s1, s2)


def jaccard_similarity_tokens(s1: str, s2: str) -> float:
    """
    Calculates the Jaccard similarity coefficient between two strings based on word tokens.
    This is useful for comparing sentences or documents.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        The Jaccard similarity coefficient (between 0 and 1).
    """
    # Split strings into word tokens and convert to sets
    tokens1 = set(s1.lower().split())
    tokens2 = set(s2.lower().split())

    # Handle the case where both sets are empty
    if len(tokens1) == 0 and len(tokens2) == 0:
        return 1.0

    # Calculate intersection and union
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    # Calculate Jaccard similarity
    similarity = len(intersection) / len(union) if len(union) > 0 else 0.0

    return similarity


def jaccard_similarity_numpy(s1: str, s2: str) -> float:
    """
    Calculates the Jaccard similarity using NumPy arrays for character-based comparison.
    This implementation converts strings to numpy arrays of unique characters.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        The Jaccard similarity coefficient (between 0 and 1).
    """
    # Convert strings to numpy arrays of unique characters
    arr1 = np.array(list(set(s1)))
    arr2 = np.array(list(set(s2)))

    # Handle empty arrays
    if len(arr1) == 0 and len(arr2) == 0:
        return 1.0

    # Calculate intersection using numpy
    intersection = np.intersect1d(arr1, arr2)

    # Calculate union using numpy
    union = np.union1d(arr1, arr2)

    # Calculate Jaccard similarity
    similarity = len(intersection) / len(union) if len(union) > 0 else 0.0

    return similarity


# Example Usage:
word1 = "intention"
word2 = "execution"
similarity = jaccard_similarity(word1, word2)
distance = jaccard_distance(word1, word2)
print(f"The Jaccard similarity between '{word1}' and '{word2}' is: {similarity:.4f}")
print(f"The Jaccard distance between '{word1}' and '{word2}' is: {distance:.4f}")

word3 = "book"
word4 = "back"
similarity2 = jaccard_similarity(word3, word4)
print(f"The Jaccard similarity between '{word3}' and '{word4}' is: {similarity2:.4f}")

# Example with sentences (token-based)
sentence1 = "the quick brown fox jumps over the lazy dog"
sentence2 = "the lazy dog sleeps under the brown tree"
token_similarity = jaccard_similarity_tokens(sentence1, sentence2)
print(f"\nToken-based Jaccard similarity between sentences: {token_similarity:.4f}")

# Example with NumPy implementation
similarity_numpy = jaccard_similarity_numpy(word1, word2)
print(f"\nNumPy-based Jaccard similarity between '{word1}' and '{word2}' is: {similarity_numpy:.4f}")


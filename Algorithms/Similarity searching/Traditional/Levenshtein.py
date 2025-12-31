"""
Levenshtein Distance Calculation using NumPy and Dynamic Programming.

The Levenshtein distance algorithm, also known as the edit distance algorithm, is a method for measuring
the similarity between two strings by calculating the minimum number of single-character edits (insertions,
deletions, or substitutions) required to change one word into the other.

The algorithm is a classic example of dynamic programming, where a matrix is used to store and build up
the edit distances between all prefixes of the two strings being compared.

Algorithm Overview:
1. Initialization: A matrix D of size (m+1) x (n+1) is created, where m and n are the lengths of the two
   strings. The first row and column are filled with values from 0 to m and 0 to n respectively. These
   represent the cost of transforming an empty string into a prefix of the other string (essentially,
   a series of insertions or deletions).

2. Matrix Filling: The matrix is filled cell by cell, typically from the top-left to the bottom-right
   corner. For each cell D[i][j], the algorithm considers the characters at the corresponding positions
   in both strings:
    * If the characters match: The cost of substitution is 0, so D[i][j] is simply the value of the
      diagonal cell D[i-1][j-1].
    * If the characters do not match: The cost is 1 (for the current operation), and D[i][j] is set to
      1 + minimum of the three adjacent cells:
        - D[i-1][j] (deletion from s1)
        - D[i][j-1] (insertion to s1)
        - D[i-1][j-1] (substitution)

3. Final Result: The value in the bottom-right corner of the matrix, D[m][n], is the final Levenshtein
   distance between the two full strings.

Properties:
1. Range: The distance ranges from 0 (identical strings) to max(len(s1), len(s2))
2. Symmetry: distance(s1, s2) = distance(s2, s1)
3. Triangle Inequality: distance(s1, s3) ≤ distance(s1, s2) + distance(s2, s3)
4. Non-negativity: distance(s1, s2) ≥ 0

Applications:
- Spell checking and auto-correction
- DNA sequence analysis and bioinformatics
- Natural language processing (fuzzy string matching)
- Plagiarism detection
- Data deduplication and record linkage
- Speech recognition
- OCR error correction

Example: "kitten" and "sitting"
To transform "kitten" into "sitting", the distance is 3:
1. kitten → sitten (substitute 'k' with 's')
2. sitten → sittin (substitute 'e' with 'i')
3. sittin → sitting (insert 'g' at the end)

Time Complexity: O(m * n) where m and n are the lengths of the two strings
Space Complexity: O(m * n) for the full matrix (can be optimized to O(min(m, n)))

"""

import numpy as np
from typing import Tuple, List


def levenshtein_distance_numpy(s1: str, s2: str) -> int:
    """
    Calculates the Levenshtein distance between two strings using NumPy for matrix operations.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        The Levenshtein distance (minimum number of edits).
    """
    m, n = len(s1), len(s2)
    # Create a matrix of zeros with dimensions (m+1) x (n+1)
    # dtype=int is important for the mathematical operations
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Initialize the first row and column
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    # Fill the matrix using dynamic programming
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Determine the cost of substitution
            cost = 0 if s1[i - 1] == s2[j - 1] else 1

            # The current cell value is the minimum of deletion, insertion, and substitution costs
            dp[i, j] = min(
                dp[i - 1, j] + 1,  # Deletion
                dp[i, j - 1] + 1,  # Insertion
                dp[i - 1, j - 1] + cost  # Substitution
            )

    # The bottom-right cell contains the final Levenshtein distance
    return dp[m, n]


# Example Usage:
word1 = "intention"
word2 = "execution"
distance = levenshtein_distance_numpy(word1, word2)
print(f"The Levenshtein distance between '{word1}' and '{word2}' is: {distance}")

word3 = "book"
word4 = "back"
distance2 = levenshtein_distance_numpy(word3, word4)
print(f"The Levenshtein distance between '{word3}' and '{word4}' is: {distance2}")

word1 = "Levenshtein"
word2 = "Livinshten"
distance = levenshtein_distance_numpy(word1, word2)
print(f"The Levenshtein distance between '{word1}' and '{word2}' is: {distance}")


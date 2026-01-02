"""Decision Tree classifier using ID3-style entropy splits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


@dataclass
class TreeNode:
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    label: Optional[int] = None

    def is_leaf(self) -> bool:
        return self.label is not None


class DecisionTree:
    """Binary decision tree using entropy/information gain splits."""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[TreeNode] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        return np.array([self._traverse(self.root, sample) for sample in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        if len(set(y)) == 1 or len(y) < self.min_samples_split or depth >= self.max_depth:
            return TreeNode(label=self._majority_class(y))

        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return TreeNode(label=self._majority_class(y))

        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature_index=feat_idx, threshold=threshold, left=left_child, right=right_child)

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[Optional[int], Optional[float]]:
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        current_entropy = self._entropy(y)

        n_samples, n_features = X.shape
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if not left_mask.any() or not right_mask.any():
                    continue

                gain = self._information_gain(y, left_mask, right_mask, current_entropy)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _entropy(self, y: np.ndarray) -> float:
        values, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _information_gain(self, y: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray, base_entropy: float) -> float:
        left_y = y[left_mask]
        right_y = y[right_mask]
        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)
        gain = base_entropy - (left_weight * self._entropy(left_y) + right_weight * self._entropy(right_y))
        return gain

    def _majority_class(self, y: np.ndarray) -> int:
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])

    def _traverse(self, node: TreeNode, sample: np.ndarray) -> int:
        if node.is_leaf():
            return node.label  # type: ignore
        if sample[node.feature_index] <= node.threshold:  # type: ignore
            return self._traverse(node.left, sample)
        return self._traverse(node.right, sample)


if __name__ == "__main__":
    X = np.array([
        [2.7, 2.5],
        [1.3, 3.1],
        [3.0, 3.0],
        [2.0, 1.0],
        [3.0, 1.1],
        [2.0, 2.0],
    ])
    y = np.array([0, 0, 1, 1, 1, 0])

    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)
    preds = tree.predict(X)
    print("Predictions:", preds)
    print("Accuracy:", np.mean(preds == y))


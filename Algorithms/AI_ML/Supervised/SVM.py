"""Linear Support Vector Machine using hinge loss + gradient descent."""

from __future__ import annotations

import numpy as np


class LinearSVM:
    def __init__(self, learning_rate: float = 0.01, reg_strength: float = 0.01, n_iters: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.reg_strength = reg_strength
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearSVM':
        X = np.array(X, dtype=float)
        y = np.where(np.array(y) <= 0, -1, 1)  # ensure labels are ±1
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.n_iters):
            margins = y * (X @ self.weights + self.bias)
            misclassified = margins < 1
            dw = self.reg_strength * self.weights - (X[misclassified] * y[misclassified][:, None]).sum(axis=0) / n_samples
            db = -(y[misclassified]).sum() / n_samples
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(self.decision_function(X))


if __name__ == "__main__":
    X = np.array([
        [2, 3],
        [1, 1],
        [2, 1],
        [2, 0],
        [0, 2],
        [1, 0],
    ])
    y = np.array([1, 1, 1, -1, -1, -1])

    svm = LinearSVM(learning_rate=0.01, reg_strength=0.01, n_iters=5000)
    svm.fit(X, y)
    preds = svm.predict(X)
    print("Predictions:", preds)


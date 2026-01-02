"""Principal Component Analysis implementation using NumPy only."""

from __future__ import annotations

import numpy as np


class PCA:
    """Compute principal components via eigen decomposition of the covariance matrix."""

    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> 'PCA':
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance matrix (features x features)
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort by eigenvalue descending
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components_ = eigenvectors[:, : self.n_components]
        self.explained_variance_ = eigenvalues[: self.n_components]
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        X_centered = X - self.mean_
        return X_centered @ self.components_

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        return X_transformed @ self.components_.T + self.mean_


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    pca = PCA(n_components=2).fit(X)

    transformed = pca.transform(X)
    reconstructed = pca.inverse_transform(transformed)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Original shape:", X.shape, "Reduced shape:", transformed.shape)


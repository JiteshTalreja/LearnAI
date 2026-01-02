"""K-Means clustering implementation using NumPy only."""
from __future__ import annotations

import numpy as np


class KMeans:
    """Classic Lloyd's algorithm for clustering."""

    def __init__(
        self,
        n_clusters: int = 3,
        max_iters: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.array(X, dtype=float)
        self._init_centroids(X)

        for _ in range(self.max_iters):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            if shift <= self.tol:
                break

        self.labels_ = labels
        distances = np.linalg.norm(X - self.centroids[self.labels_], axis=1)
        self.inertia_ = float(np.sum(distances ** 2))
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _init_centroids(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), size=self.n_clusters, replace=False)
        self.centroids = X[indices]

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.n_clusters):
            members = X[labels == k]
            new_centroids[k] = members.mean(axis=0) if len(members) else self.centroids[k]
        return new_centroids


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cluster1 = rng.normal(loc=(0, 0), scale=0.3, size=(50, 2))
    cluster2 = rng.normal(loc=(3, 3), scale=0.3, size=(50, 2))
    cluster3 = rng.normal(loc=(0, 4), scale=0.3, size=(50, 2))
    data = np.vstack([cluster1, cluster2, cluster3])

    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(data)

    print("Inertia:", round(model.inertia_, 3))
    print("Centroids:\n", model.centroids)
    print("Unique labels:", np.unique(labels))

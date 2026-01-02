"""
K-Nearest Neighbors (KNN) Algorithm Implementation using NumPy.

K-Nearest Neighbors is a simple, instance-based learning algorithm that classifies new data points
based on the majority class of their K nearest neighbors in the feature space. It's a "lazy learning"
algorithm because it doesn't learn a discriminative function but instead memorizes the training dataset.

Algorithm Overview:
1. Store Training Data: No actual training phase, just store all data points with their labels
2. Calculate Distances: For a new query point, calculate distance to all training points
3. Find K Neighbors: Select the K points with smallest distances
4. Vote: Return the majority class among K neighbors (classification) or average (regression)

Distance Metrics:
- Euclidean: d(x,y) = √(Σ(xᵢ - yᵢ)²) - Most common, works well for continuous features
- Manhattan: d(x,y) = Σ|xᵢ - yᵢ| - Better for high-dimensional data
- Minkowski: d(x,y) = (Σ|xᵢ - yᵢ|^p)^(1/p) - Generalized form (p=2 is Euclidean, p=1 is Manhattan)

Choosing K:
- Small K (e.g., 1-3): More sensitive to noise, complex decision boundary
- Large K: Smoother boundary, may miss local patterns
- Common heuristic: K = √n where n is the number of training samples
- Best practice: Use cross-validation to find optimal K

Properties:
- Non-parametric: Makes no assumptions about data distribution
- Lazy Learning: All computation happens at prediction time
- Time Complexity: O(1) for training, O(n × d) for each prediction
- Space Complexity: O(n × d) to store all training data

Applications:
- Image recognition and classification
- Recommendation systems (collaborative filtering)
- Medical diagnosis (disease classification)
- Handwriting recognition
- Pattern recognition and anomaly detection
- Credit scoring

Advantages:
- Simple to understand and implement
- No training phase needed
- Naturally handles multi-class problems
- Can be used for both classification and regression

Disadvantages:
- Slow prediction for large datasets
- Sensitive to irrelevant features
- Requires feature scaling
- Memory-intensive (stores all training data)
"""

import numpy as np
from collections import Counter
from typing import List, Union, Literal


class KNN:
    """
    K-Nearest Neighbors classifier implementation from scratch using NumPy.

    This implementation supports multiple distance metrics and can be used
    for both classification tasks with weighted or unweighted voting.
    """

    def __init__(
        self,
        k: int = 3,
        distance_metric: Literal['euclidean', 'manhattan', 'minkowski'] = 'euclidean',
        p: int = 2,
        weighted: bool = False
    ):
        """
        Initialize KNN classifier.

        Args:
            k: Number of neighbors to consider for voting
            distance_metric: Distance metric to use ('euclidean', 'manhattan', 'minkowski')
            p: Power parameter for Minkowski distance (p=2 is Euclidean, p=1 is Manhattan)
            weighted: If True, closer neighbors have more influence on the vote
        """
        self.k = k
        self.distance_metric = distance_metric
        self.p = p
        self.weighted = weighted
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.

        Formula: d(x,y) = √(Σ(xᵢ - yᵢ)²)
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Manhattan distance between two points.

        Formula: d(x,y) = Σ|xᵢ - yᵢ|
        """
        return np.sum(np.abs(x1 - x2))

    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Minkowski distance between two points.

        Formula: d(x,y) = (Σ|xᵢ - yᵢ|^p)^(1/p)
        """
        return np.power(np.sum(np.abs(x1 - x2) ** self.p), 1 / self.p)

    def _get_distance_function(self):
        """Return the appropriate distance function based on the metric."""
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance
        elif self.distance_metric == 'minkowski':
            return self._minkowski_distance
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def fit(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> 'KNN':
        """
        Fit the KNN model (store training data).

        KNN is a lazy learner, so this just stores the training data.

        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)

        Returns:
            self: The fitted KNN instance
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def _predict_single(self, x: np.ndarray) -> Union[int, float]:
        """
        Predict the label for a single sample.

        Args:
            x: Single sample of shape (n_features,)

        Returns:
            Predicted label
        """
        distance_func = self._get_distance_function()

        # Calculate distances to all training points
        distances = np.array([distance_func(x, x_train) for x_train in self.X_train])

        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # Get labels of k nearest neighbors
        k_labels = self.y_train[k_indices]

        if self.weighted:
            # Weighted voting: closer neighbors have more influence
            k_distances = distances[k_indices]
            # Add small epsilon to avoid division by zero
            weights = 1 / (k_distances + 1e-10)

            # Weighted vote
            label_weights = {}
            for label, weight in zip(k_labels, weights):
                label_weights[label] = label_weights.get(label, 0) + weight

            return max(label_weights, key=label_weights.get)
        else:
            # Simple majority voting
            most_common = Counter(k_labels).most_common(1)
            return most_common[0][0]

    def predict(self, X: Union[List, np.ndarray]) -> np.ndarray:
        """
        Predict labels for multiple samples.

        Args:
            X: Test features of shape (n_samples, n_features)

        Returns:
            Predicted labels of shape (n_samples,)
        """
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X: Union[List, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for samples.

        Args:
            X: Test features of shape (n_samples, n_features)

        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        X = np.array(X)
        distance_func = self._get_distance_function()
        probabilities = []

        for x in X:
            # Calculate distances to all training points
            distances = np.array([distance_func(x, x_train) for x_train in self.X_train])

            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]

            # Calculate probability for each class
            probs = np.zeros(len(self.classes_))
            for i, cls in enumerate(self.classes_):
                probs[i] = np.sum(k_labels == cls) / self.k

            probabilities.append(probs)

        return np.array(probabilities)

    def score(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> float:
        """
        Calculate classification accuracy.

        Args:
            X: Test features of shape (n_samples, n_features)
            y: True labels of shape (n_samples,)

        Returns:
            Accuracy score (fraction of correct predictions)
        """
        predictions = self.predict(X)
        return np.mean(predictions == np.array(y))

    def get_neighbors(self, x: Union[List, np.ndarray]) -> tuple:
        """
        Get the K nearest neighbors for a query point.

        Args:
            x: Query point of shape (n_features,)

        Returns:
            Tuple of (neighbor_indices, neighbor_distances, neighbor_labels)
        """
        x = np.array(x)
        distance_func = self._get_distance_function()

        distances = np.array([distance_func(x, x_train) for x_train in self.X_train])
        k_indices = np.argsort(distances)[:self.k]

        return k_indices, distances[k_indices], self.y_train[k_indices]


class KNNRegressor:
    """
    K-Nearest Neighbors regressor for continuous target prediction.

    Instead of voting for a class, it returns the average (or weighted average)
    of the K nearest neighbors' target values.
    """

    def __init__(self, k: int = 3, weighted: bool = False):
        """
        Initialize KNN regressor.

        Args:
            k: Number of neighbors to consider
            weighted: If True, use distance-weighted average
        """
        self.k = k
        self.weighted = weighted
        self.X_train = None
        self.y_train = None

    def fit(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> 'KNNRegressor':
        """Store training data."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _predict_single(self, x: np.ndarray) -> float:
        """Predict value for a single sample."""
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_values = self.y_train[k_indices]

        if self.weighted:
            k_distances = distances[k_indices]
            weights = 1 / (k_distances + 1e-10)
            return np.sum(weights * k_values) / np.sum(weights)
        else:
            return np.mean(k_values)

    def predict(self, X: Union[List, np.ndarray]) -> np.ndarray:
        """Predict values for multiple samples."""
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def score(self, X: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> float:
        """Calculate R² score."""
        predictions = self.predict(X)
        y = np.array(y)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# Example Usage
if __name__ == "__main__":
    print("=" * 60)
    print("K-Nearest Neighbors (KNN) Algorithm Demo")
    print("=" * 60)

    # Classification Example
    print("\n1. CLASSIFICATION EXAMPLE")
    print("-" * 40)

    # Create sample dataset (two clusters)
    X_train = [
        [1, 2], [1.5, 1.8], [2, 2.5], [2.5, 2],  # Class 0 cluster
        [6, 5], [7, 7], [8, 6], [7.5, 5.5]        # Class 1 cluster
    ]
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]

    # Test points
    X_test = [[2, 2], [7, 6], [4, 4]]  # Near class 0, near class 1, middle
    y_test = [0, 1, 0]  # Expected labels

    # Create and train KNN classifier
    knn = KNN(k=3, distance_metric='euclidean')
    knn.fit(X_train, y_train)

    # Predictions
    predictions = knn.predict(X_test)
    probabilities = knn.predict_proba(X_test)
    accuracy = knn.score(X_test, y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"K (neighbors): {knn.k}")
    print(f"Distance metric: {knn.distance_metric}")
    print(f"\nTest Points: {X_test}")
    print(f"Predictions: {predictions}")
    print(f"True Labels: {y_test}")
    print(f"Probabilities:\n{probabilities}")
    print(f"Accuracy: {accuracy:.2%}")

    # Show neighbors for a specific point
    print(f"\nNeighbors for point [2, 2]:")
    indices, distances, labels = knn.get_neighbors([2, 2])
    for i, (idx, dist, label) in enumerate(zip(indices, distances, labels)):
        print(f"  Neighbor {i+1}: Point {X_train[idx]}, Distance: {dist:.3f}, Label: {label}")

    # Weighted KNN example
    print("\n2. WEIGHTED KNN EXAMPLE")
    print("-" * 40)

    knn_weighted = KNN(k=3, weighted=True)
    knn_weighted.fit(X_train, y_train)
    predictions_weighted = knn_weighted.predict(X_test)

    print(f"Unweighted predictions: {predictions}")
    print(f"Weighted predictions: {predictions_weighted}")

    # Regression Example
    print("\n3. REGRESSION EXAMPLE")
    print("-" * 40)

    # Simple regression dataset
    X_reg_train = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    y_reg_train = [2.1, 4.0, 6.2, 8.1, 10.0, 11.9, 14.2, 16.0, 18.1, 20.0]  # y ≈ 2x

    X_reg_test = [[2.5], [5.5], [8.5]]
    y_reg_test = [5.0, 11.0, 17.0]

    knn_reg = KNNRegressor(k=2)
    knn_reg.fit(X_reg_train, y_reg_train)

    reg_predictions = knn_reg.predict(X_reg_test)
    r2 = knn_reg.score(X_reg_test, y_reg_test)

    print(f"Test inputs: {[x[0] for x in X_reg_test]}")
    print(f"Predictions: {[f'{p:.2f}' for p in reg_predictions]}")
    print(f"True values: {y_reg_test}")
    print(f"R² Score: {r2:.4f}")

    # Different distance metrics comparison
    print("\n4. DISTANCE METRICS COMPARISON")
    print("-" * 40)

    metrics = ['euclidean', 'manhattan', 'minkowski']
    for metric in metrics:
        knn_metric = KNN(k=3, distance_metric=metric, p=3)
        knn_metric.fit(X_train, y_train)
        acc = knn_metric.score(X_test, y_test)
        print(f"{metric.capitalize():12} Accuracy: {acc:.2%}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


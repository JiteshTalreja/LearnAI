"""Linear Regression implementation using gradient descent (NumPy only)."""

import numpy as np


class LinearRegression:
    """Ordinary Least Squares solved via batch gradient descent."""

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            # Mean gradients
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float)
        return X @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_true = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot


if __name__ == "__main__":
    # Simple demo: y = 2x + 1 with noise
    rng = np.random.default_rng(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 2 * X.squeeze() + 1 + rng.normal(0, 1, size=100)

    model = LinearRegression(learning_rate=0.01, n_iterations=2000)
    model.fit(X, y)

    r2 = model.score(X, y)
    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("R^2:", round(r2, 4))


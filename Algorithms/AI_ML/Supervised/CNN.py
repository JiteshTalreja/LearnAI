"""
Convolutional Neural Network (CNN) implementation using NumPy only.

This simplified CNN is designed for small grayscale images and educational use. It implements the
core building blocks of modern convolutional models without relying on deep learning frameworks:

1. Convolution Layer (valid padding, stride 1)
2. ReLU activation
3. Max Pooling layer
4. Flatten layer
5. Dense (fully-connected) layer with softmax output

The goal is to show how convolutional feature extraction, pooling, and classification can be wired
together from scratch. This mirrors the explanatory approach used in other algorithm folders.

Example Workflow:
- Instantiate SimpleCNN with desired filter counts and target classes
- Call `forward(image)` to get class probabilities
- Call `predict(image)` to get the winning class index

Note: Training a CNN from scratch via backpropagation is outside the scope here; the implementation
focuses on the forward pass for intuition, using randomly initialized filters. You can plug in
hand-crafted filters or pre-trained weights if desired.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class ConvLayer:
    """Single convolutional layer supporting multiple filters."""

    def __init__(self, num_filters: int = 8, filter_size: int = 3) -> None:
        self.num_filters = num_filters
        self.filter_size = filter_size
        # He-style initialization keeps activations in a reasonable range
        scale = math.sqrt(2.0 / (filter_size * filter_size))
        self.filters = np.random.randn(num_filters, filter_size, filter_size) * scale

    def iterate_regions(self, image: np.ndarray):
        """Yield sub-regions of the image and their top-left coordinates."""
        h, w = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                region = image[i : i + self.filter_size, j : j + self.filter_size]
                yield region, i, j

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Perform a valid convolution over the input image."""
        h, w = image.shape
        out_h = h - self.filter_size + 1
        out_w = w - self.filter_size + 1
        output = np.zeros((out_h, out_w, self.num_filters))

        for region, i, j in self.iterate_regions(image):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))
        return output


class ReLULayer:
    """Applies the rectified linear unit activation element-wise."""

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)


class MaxPoolLayer:
    """Non-overlapping max pooling layer."""

    def __init__(self, pool_size: int = 2) -> None:
        self.pool_size = pool_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        h, w, depth = x.shape
        out_h = h // self.pool_size
        out_w = w // self.pool_size
        output = np.zeros((out_h, out_w, depth))

        for i in range(out_h):
            for j in range(out_w):
                region = x[
                    i * self.pool_size : (i + 1) * self.pool_size,
                    j * self.pool_size : (j + 1) * self.pool_size,
                ]
                output[i, j] = np.max(region, axis=(0, 1))
        return output


class FlattenLayer:
    """Flattens 3D feature maps into a 1D vector."""

    @staticmethod
    def forward(x: np.ndarray) -> np.ndarray:
        return x.reshape(-1)


class DenseLayer:
    """Simple fully-connected layer with optional softmax output."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        limit = math.sqrt(6 / (input_dim + output_dim))
        self.weights = np.random.uniform(-limit, limit, size=(input_dim, output_dim))
        self.bias = np.zeros(output_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights + self.bias
        exp = np.exp(logits - np.max(logits))
        return exp / np.sum(exp)


class SimpleCNN:
    """Convenience wrapper wiring conv, activation, pooling, and dense layers."""

    def __init__(
        self,
        image_shape: Tuple[int, int] = (28, 28),
        num_filters: int = 8,
        filter_size: int = 3,
        pool_size: int = 2,
        num_classes: int = 10,
    ) -> None:
        self.image_shape = image_shape
        self.conv = ConvLayer(num_filters=num_filters, filter_size=filter_size)
        self.relu = ReLULayer()
        self.pool = MaxPoolLayer(pool_size=pool_size)
        pooled_h = (image_shape[0] - filter_size + 1) // pool_size
        pooled_w = (image_shape[1] - filter_size + 1) // pool_size
        flattened_dim = pooled_h * pooled_w * num_filters
        self.flatten = FlattenLayer()
        self.classifier = DenseLayer(flattened_dim, num_classes)

    def forward(self, image: np.ndarray) -> np.ndarray:
        if image.shape != self.image_shape:
            raise ValueError(
                f"Expected image shape {self.image_shape}, got {image.shape}"
            )
        # Normalize to [0, 1] for stability if needed
        if image.max() > 1:
            image = image / 255.0
        features = self.conv.forward(image)
        features = self.relu.forward(features)
        features = self.pool.forward(features)
        flat = self.flatten.forward(features)
        return self.classifier.forward(flat)

    def predict(self, image: np.ndarray) -> int:
        probs = self.forward(image)
        return int(np.argmax(probs))

    def summary(self) -> None:
        print("SimpleCNN Summary")
        print(f"Input shape: {self.image_shape}")
        print(f"Conv filters: {self.conv.num_filters} x {self.conv.filter_size}x{self.conv.filter_size}")
        pooled_h = (self.image_shape[0] - self.conv.filter_size + 1) // self.pool.pool_size
        pooled_w = (self.image_shape[1] - self.conv.filter_size + 1) // self.pool.pool_size
        print(f"After pooling: {pooled_h} x {pooled_w} x {self.conv.num_filters}")
        print(f"Dense output classes: {self.classifier.bias.size}")


if __name__ == "__main__":
    np.random.seed(42)
    cnn = SimpleCNN(image_shape=(28, 28), num_filters=6, num_classes=10)
    cnn.summary()

    dummy_image = np.random.randint(0, 255, size=(28, 28)).astype(np.float32)
    probs = cnn.forward(dummy_image)
    print("Class probabilities:", np.round(probs, 3))
    print("Predicted class:", cnn.predict(dummy_image))


# AI/ML Algorithms

This folder contains implementations of fundamental AI/ML algorithms from scratch using NumPy. Each algorithm is implemented with detailed documentation explaining the mathematical concepts and practical applications.

## Table of Contents

### Supervised Learning
Algorithms that learn from labeled training data to make predictions.

| Algorithm | Description | File |
|-----------|-------------|------|
| K-Nearest Neighbors (KNN) | Instance-based classification/regression | `Supervised/KNN.py` |
| Convolutional Neural Network (CNN) | Deep learning for image processing | `Supervised/CNN.py` |
| Linear Regression | Predicting continuous values | `Supervised/LinearRegression.py` |
| Support Vector Machine (SVM) | Max-margin linear classifier | `Supervised/SVM.py` |
| Decision Tree | Rule-based classifier using entropy gain | `Supervised/DecisionTree.py` |

### Unsupervised Learning
Algorithms that find patterns in unlabeled data.

| Algorithm | Description | File |
|-----------|-------------|------|
| K-Means Clustering | Partition data into K clusters | `Unsupervised/KMeans.py` |
| PCA | Dimensionality reduction | `Unsupervised/PCA.py` |

---

## Overview

### What is Machine Learning?

Machine Learning is a subset of Artificial Intelligence that enables systems to learn and improve from experience without being explicitly programmed. The algorithms in this folder are categorized based on their learning paradigm:

**Supervised Learning:**
- Uses labeled training data
- Learns a mapping from inputs to outputs
- Examples: Classification, Regression

**Unsupervised Learning:**
- Uses unlabeled data
- Discovers hidden patterns or structures
- Examples: Clustering, Dimensionality Reduction

---

## Quick Start

```python
# Example: Using KNN for classification
from Supervised.KNN import KNN

# Training data
X_train = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]]
y_train = [0, 0, 0, 1, 1, 1]

# Create and train model
knn = KNN(k=3)
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict([[2, 2], [7, 6]])
print(predictions)  # [0, 1]
```

```python
# Example: Using K-Means for clustering
from Unsupervised.KMeans import KMeans

# Data points
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]

# Create and fit model
kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(X)
print(labels)  # Cluster assignments
```

---

## Dependencies

All implementations use only NumPy for numerical operations:

```bash
pip install numpy
```

---

## Folder Structure

```
AI_ML/
├── README.md                    # This file
├── Supervised/
│   ├── README.md               # Supervised learning algorithms documentation
│   ├── KNN.py                  # K-Nearest Neighbors
│   ├── CNN.py                  # Convolutional Neural Network
│   ├── LinearRegression.py     # Linear Regression
│   ├── SVM.py                  # Support Vector Machine
│   └── DecisionTree.py         # Decision Tree
└── Unsupervised/
    ├── README.md               # Unsupervised learning algorithms documentation
    ├── KMeans.py               # K-Means Clustering
    └── PCA.py                  # Principal Component Analysis
```

# Supervised Learning Algorithms

This folder contains implementations of supervised learning algorithms. Supervised learning uses labeled training data to learn a mapping function from inputs to outputs.

## Table of Contents
- [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
- [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
- [Linear Regression](#linear-regression)
- [Support Vector Machine (SVM)](#support-vector-machine-svm)
- [Decision Tree](#decision-tree)

---

## K-Nearest Neighbors (KNN)

### Algorithm Overview

**K-Nearest Neighbors (KNN)** is a simple, instance-based learning algorithm that classifies new data points based on the majority class of their K nearest neighbors in the feature space.

**Mathematical Definition:**

For a query point x, KNN finds the K closest training points and assigns the most frequent class:

```
ŷ = mode({y_i : x_i ∈ N_k(x)})
```

Where N_k(x) is the set of K nearest neighbors of x.

**Distance Metrics:**

1. **Euclidean Distance** (most common):
   ```
   d(x, y) = √(Σ(x_i - y_i)²)
   ```

2. **Manhattan Distance**:
   ```
   d(x, y) = Σ|x_i - y_i|
   ```

3. **Minkowski Distance** (generalized):
   ```
   d(x, y) = (Σ|x_i - y_i|^p)^(1/p)
   ```

**Algorithm Steps:**

1. **Store Training Data**: No actual training, just store all data points
2. **Calculate Distances**: For a new query point, calculate distance to all training points
3. **Find K Neighbors**: Select the K points with smallest distances
4. **Vote**: Return the majority class among K neighbors (classification) or average value (regression)

**Example:**
```
Training Data:
Point A (1, 2) → Class 0
Point B (2, 3) → Class 0
Point C (6, 5) → Class 1
Point D (7, 7) → Class 1

Query Point: (2, 2) with K=3

Distances:
- To A: √((2-1)² + (2-2)²) = 1.0
- To B: √((2-2)² + (2-3)²) = 1.0
- To C: √((2-6)² + (2-5)²) = 5.0
- To D: √((2-7)² + (2-7)²) = 7.07

K=3 Nearest: A(Class 0), B(Class 0), C(Class 1)
Majority Vote: Class 0
```

**Properties:**
- **Lazy Learning**: No training phase, all computation at prediction time
- **Non-parametric**: Makes no assumptions about data distribution
- **Time Complexity**: O(1) training, O(n × d) prediction (n=samples, d=features)
- **Space Complexity**: O(n × d) to store training data

**Choosing K:**
- Small K: More sensitive to noise, complex decision boundary
- Large K: Smoother boundary, may miss local patterns
- Common practice: K = √n or use cross-validation

**Applications:**
- Image recognition
- Recommendation systems
- Medical diagnosis
- Handwriting recognition
- Pattern recognition

### Code Implementation

**File:** `KNN.py`

The implementation provides:

1. **`KNN.__init__(k, distance_metric)`**:
   - Initialize with number of neighbors and distance metric
   - Supports 'euclidean', 'manhattan', 'minkowski' distances

2. **`KNN.fit(X, y)`**:
   - Store training data (no actual training)

3. **`KNN.predict(X)`**:
   - Predict labels for new data points
   - Returns array of predicted labels

4. **`KNN.predict_proba(X)`**:
   - Returns probability distribution over classes

5. **`KNN.score(X, y)`**:
   - Calculate accuracy on test data

**Key Features:**
- Multiple distance metrics supported
- Weighted voting option (closer neighbors have more influence)
- Both classification and regression support
- Efficient NumPy vectorized operations

**Example Usage:**
```python
from KNN import KNN

# Create and train
knn = KNN(k=3, distance_metric='euclidean')
knn.fit(X_train, y_train)

# Predict
predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

---

## Convolutional Neural Network (CNN)

### Algorithm Overview

**Convolutional Neural Networks (CNNs)** are deep learning models specifically designed for processing structured grid data like images. They use convolution operations to automatically learn spatial hierarchies of features.

**Key Components:**

1. **Convolution Layer**: Applies learnable filters to extract features
2. **Activation Function (ReLU)**: Introduces non-linearity
3. **Pooling Layer**: Reduces spatial dimensions
4. **Fully Connected Layer**: Final classification

**Mathematical Operations:**

**Convolution:**
```
(I * K)[i,j] = ΣΣ I[i+m, j+n] × K[m, n]
```
Where I is the input image and K is the kernel/filter.

**ReLU Activation:**
```
f(x) = max(0, x)
```

**Max Pooling:**
```
output[i,j] = max(input[i*s:(i+1)*s, j*s:(j+1)*s])
```
Where s is the stride/pool size.

**Softmax:**
```
softmax(z_i) = e^(z_i) / Σe^(z_j)
```

**Algorithm Steps:**

1. **Convolution**: Slide filters across input, compute dot products
2. **Activation**: Apply ReLU to introduce non-linearity
3. **Pooling**: Downsample to reduce dimensions and computation
4. **Flatten**: Convert 3D feature maps to 1D vector
5. **Fully Connected**: Dense layers for final classification
6. **Softmax**: Output probability distribution over classes

**Example:**
```
Input Image: 28×28 grayscale

Convolution (8 filters, 3×3):
- Output: 26×26×8

ReLU:
- Output: 26×26×8 (negative values → 0)

Max Pooling (2×2):
- Output: 13×13×8

Flatten:
- Output: 1352 (13×13×8)

Fully Connected:
- Output: 10 classes (probabilities)
```

**Properties:**
- **Parameter Sharing**: Same filter used across entire image
- **Local Connectivity**: Each neuron connects only to local region
- **Translation Invariance**: Can detect features regardless of position
- **Hierarchical Features**: Lower layers learn edges, higher layers learn complex patterns

**Applications:**
- Image classification
- Object detection
- Facial recognition
- Medical image analysis
- Self-driving cars

### Code Implementation

**File:** `CNN.py`

The implementation provides:

1. **`ConvLayer`**: Convolutional layer with learnable filters
2. **`MaxPoolLayer`**: Max pooling for downsampling
3. **`ReLU`**: ReLU activation function
4. **`SoftmaxLayer`**: Fully connected layer with softmax
5. **`SimpleCNN`**: Complete CNN combining all layers

**Key Features:**
- Forward propagation implemented from scratch
- Backpropagation with gradient descent
- Configurable number of filters and sizes
- Support for training and inference modes

**Example Usage:**
```python
from CNN import SimpleCNN

# Create CNN
cnn = SimpleCNN(num_filters=8, filter_size=3, num_classes=10)

# Predict (28×28 image)
prediction = cnn.predict(image)
print(f"Predicted class: {prediction}")
```

---

## Linear Regression

### Algorithm Overview

**Linear Regression** finds the best-fitting linear relationship between input features and a continuous target variable by minimizing the sum of squared residuals.

**Mathematical Definition:**

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = Xw
```

**Cost Function (Mean Squared Error):**
```
J(w) = (1/2n) × Σ(ŷᵢ - yᵢ)²
```

**Gradient Descent Update:**
```
w = w - α × ∇J(w)
∇J(w) = (1/n) × Xᵀ(Xw - y)
```

**Closed-Form Solution (Normal Equation):**
```
w = (X��X)⁻¹Xᵀy
```

**Algorithm Steps:**

1. **Initialize Weights**: Start with random or zero weights
2. **Compute Predictions**: ŷ = Xw
3. **Calculate Loss**: MSE between predictions and actual values
4. **Compute Gradients**: Partial derivatives of loss w.r.t. weights
5. **Update Weights**: Move in opposite direction of gradient
6. **Repeat**: Until convergence or max iterations

**Properties:**
- **Linear Model**: Assumes linear relationship between features and target
- **Closed-form Solution**: Can be solved analytically
- **Interpretable**: Weights show feature importance
- **Time Complexity**: O(n × d²) for normal equation, O(n × d × iterations) for gradient descent

**Applications:**
- House price prediction
- Sales forecasting
- Risk assessment
- Trend analysis

### Code Implementation

**File:** `LinearRegression.py`

The implementation provides:

1. **`LinearRegression.fit(X, y)`**: Train using gradient descent or normal equation
2. **`LinearRegression.predict(X)`**: Predict continuous values
3. **`LinearRegression.score(X, y)`**: Calculate R² score

**Example Usage:**
```python
from LinearRegression import LinearRegression

# Create and train
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
r2_score = model.score(X_test, y_test)
print(f"R² Score: {r2_score:.4f}")
```

---

## Decision Tree

### Algorithm Overview
Decision Trees split the feature space recursively by choosing attributes that maximize information gain (or minimize Gini impurity). Each internal node represents a test on a feature, each branch corresponds to an outcome, and each leaf stores a class label.

**Entropy:**
```
H(S) = - �� p_i log2 p_i
```

**Information Gain:**
```
Gain(S, A) = H(S) - Σ (|S_v| / |S|) * H(S_v)
```

**Algorithm Steps:**
1. If all samples share the same label, return a leaf.
2. Otherwise, for each feature compute information gain.
3. Split on the feature with highest gain.
4. Recurse on child subsets until max depth or pure leaves.

**Properties:**
- Non-linear decision boundaries
- Easily interpretable rules
- Prone to overfitting (controlled via max depth / min samples)

### Code Implementation
- **File:** `DecisionTree.py`
- **Features:** ID3-style entropy gain, max depth, min samples split, predict method for single/multiple samples.

---

## Support Vector Machine (SVM)

### Algorithm Overview
SVM finds the hyperplane that maximizes the margin between two classes. For linearly separable data, this corresponds to solving an optimization problem that balances margin size and classification errors.

**Primal Form:**
```
minimize  1/2 ||w||^2 + C Σ ξ_i
subject to y_i (w · x_i + b) ≥ 1 - ξ_i,  ξ_i ≥ 0
```

In this simplified implementation we focus on the hard-margin/separable case, solved via gradient descent on the hinge loss.

**Hinge Loss:**
```
L = Σ max(0, 1 - y_i (w · x_i + b)) + λ ||w||^2
```

**Algorithm Steps:**
1. Initialize weights and bias
2. For each iteration, compute gradients of hinge loss
3. Update parameters via gradient descent

### Code Implementation
- **File:** `SVM.py`
- **Features:** Linear SVM using hinge loss + L2 regularization, supports batch training, predicts class labels/signs, exposes decision function.

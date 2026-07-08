import numpy as np


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

def relu(Z: np.ndarray) -> np.ndarray:
    """
    ReLU activation:  A = max(0, Z)

    Parameters
    ----------
    Z : np.ndarray, shape (n, m)

    Returns
    -------
    A : np.ndarray, same shape as Z
    """
    A = np.maximum(0, Z)
    return A


def relu_derivative(Z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU:  1 if Z > 0, else 0.

    Parameters
    ----------
    Z : np.ndarray, shape (n, m)

    Returns
    -------
    dA : np.ndarray, same shape as Z
        Element-wise derivative.
    """
    return (Z > 0).astype(np.float32)


def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Softmax activation (stable version).

    A_i = exp(Z_i) / Σ exp(Z_j)   (computed per column)

    Parameters
    ----------
    Z : np.ndarray, shape (num_classes, m)

    Returns
    -------
    A : np.ndarray, same shape as Z
        Each column sums to 1.

    Hint
    ----
    For numerical stability, subtract max(Z) per column before exp.
    """
    # Subtract max(Z) per column for numerical stability
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    return A


# ============================================================
# PARAMETER INITIALIZATION
# ============================================================

def initialize_parameters(layer_dims: list[int]) -> dict:
    """
    Initialize weights and biases for an L-layer network.

    Parameters
    ----------
    layer_dims : list of int
        [n_input, n_hidden_1, ..., n_hidden_L-1, n_output]
        e.g. [784, 128, 64, 10]

    Returns
    -------
    parameters : dict
        {
            "W1": shape (layer_dims[1], layer_dims[0]),
            "b1": shape (layer_dims[1], 1),
            "W2": shape (layer_dims[2], layer_dims[1]),
            "b2": shape (layer_dims[2], 1),
            ...
        }

    Hints
    -----
    - Use He initialization for ReLU layers:
        W[l] = np.random.randn(...) * sqrt(2 / layer_dims[l-1])
    - Initialize biases to zeros.
    """
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        # He initialization for weights
        parameters["W" + str(l)] = (np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])).astype(np.float32)
        # Initialize biases to zeros
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1), dtype=np.float32)
        
    return parameters


# ============================================================
# FORWARD PROPAGATION
# ============================================================

def forward_propagation(X: np.ndarray, parameters: dict) -> tuple[np.ndarray, dict]:
    """
    Full forward pass through the L-layer network.

    Architecture: [LINEAR → RELU] × (L-1)  →  [LINEAR → SOFTMAX]
    (All hidden layers use ReLU; the output layer uses Softmax.)

    Parameters
    ----------
    X : np.ndarray, shape (784, m)
        Input data (each column is one example).
    parameters : dict
        Weights and biases from initialize_parameters().

    Returns
    -------
    AL : np.ndarray, shape (num_classes, m)
        Output probabilities from the softmax layer.
    cache : dict
        Store ALL intermediate values needed for backprop:
        {
            "Z1": ..., "A0": X, "A1": ...,
            "Z2": ..., "A2": ...,
            ...
        }
        You'll need Z[l] and A[l-1] for every layer during backprop.

    Math per layer l
    ----------------
    Z[l] = W[l] · A[l-1] + b[l]
    A[l] = activation(Z[l])
    """
    cache = {"A0": X}
    L = len(parameters) // 2
    
    for l in range(1, L):
        W_l = parameters["W" + str(l)]
        b_l = parameters["b" + str(l)]
        A_prev = cache["A" + str(l-1)]
        
        Z_l = np.dot(W_l, A_prev) + b_l
        A_l = relu(Z_l)
        
        cache["Z" + str(l)] = Z_l
        cache["A" + str(l)] = A_l
        
    # Final layer (Softmax)
    W_L = parameters["W" + str(L)]
    b_L = parameters["b" + str(L)]
    A_prev = cache["A" + str(L-1)]
    
    Z_L = np.dot(W_L, A_prev) + b_L
    AL = softmax(Z_L)
    
    cache["Z" + str(L)] = Z_L
    cache["A" + str(L)] = AL
    
    return AL, cache


# ============================================================
# COST FUNCTION
# ============================================================

def compute_cost(AL: np.ndarray, Y_one_hot: np.ndarray) -> float:
    """
    Cross-entropy cost.

    L = - (1/m) Σ Σ Y_ij * log(AL_ij)

    Parameters
    ----------
    AL : np.ndarray, shape (num_classes, m)
        Predicted probabilities.
    Y_one_hot : np.ndarray, shape (num_classes, m)
        True labels (one-hot).

    Returns
    -------
    cost : float
        Scalar cross-entropy cost.

    Hint
    ----
    Add a tiny epsilon (1e-8) inside log() to avoid log(0).
    """
    m = AL.shape[1]
    cost = - (1.0 / m) * np.sum(Y_one_hot * np.log(AL + 1e-8))
    return float(np.squeeze(cost))


# ============================================================
# BACKWARD PROPAGATION
# ============================================================

def backward_propagation(AL: np.ndarray, Y_one_hot: np.ndarray,
                         parameters: dict, cache: dict) -> dict:
    """
    Full backward pass through the L-layer network.

    Parameters
    ----------
    AL : np.ndarray, shape (num_classes, m)
        Output of forward_propagation (softmax probabilities).
    Y_one_hot : np.ndarray, shape (num_classes, m)
        True labels (one-hot encoded).
    parameters : dict
        Current weights and biases.
    cache : dict
        Intermediate values from forward_propagation.

    Returns
    -------
    grads : dict
        {
            "dW1": ..., "db1": ...,
            "dW2": ..., "db2": ...,
            ...
        }

    Math
    ----
    Output layer (softmax + cross-entropy combined):
        dZ[L] = AL - Y_one_hot

    Hidden layers (ReLU):
        dA[l]   = W[l+1]ᵀ · dZ[l+1]
        dZ[l]   = dA[l] * relu_derivative(Z[l])

    For every layer:
        dW[l] = (1/m) · dZ[l] · A[l-1]ᵀ
        db[l] = (1/m) · Σ dZ[l]  (sum over columns, keepdims=True)
    """
    grads = {}
    L = len(parameters) // 2
    m = AL.shape[1]
    
    # 1. Output layer L (Softmax + Cross-Entropy)
    dZ_curr = AL - Y_one_hot
    grads["dW" + str(L)] = (1.0 / m) * np.dot(dZ_curr, cache["A" + str(L-1)].T)
    grads["db" + str(L)] = (1.0 / m) * np.sum(dZ_curr, axis=1, keepdims=True)
    
    # 2. Hidden layers (L-1 down to 1)
    for l in range(L - 1, 0, -1):
        W_next = parameters["W" + str(l+1)]
        dA_l = np.dot(W_next.T, dZ_curr)
        
        Z_l = cache["Z" + str(l)]
        dZ_l = dA_l * relu_derivative(Z_l)
        
        A_prev = cache["A" + str(l-1)]
        grads["dW" + str(l)] = (1.0 / m) * np.dot(dZ_l, A_prev.T)
        grads["db" + str(l)] = (1.0 / m) * np.sum(dZ_l, axis=1, keepdims=True)
        
        dZ_curr = dZ_l
        
    return grads


# ============================================================
# PARAMETER UPDATE
# ============================================================

def update_parameters(parameters: dict, grads: dict,
                      learning_rate: float) -> dict:
    """
    Gradient descent update.

    W[l] = W[l] - α · dW[l]
    b[l] = b[l] - α · db[l]

    Parameters
    ----------
    parameters : dict
    grads : dict
    learning_rate : float (α)

    Returns
    -------
    parameters : dict   (updated in-place is fine)
    """
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters


# ============================================================
# PREDICTION
# ============================================================

def predict(X: np.ndarray, parameters: dict) -> np.ndarray:
    """
    Run forward prop and return predicted class labels.

    Parameters
    ----------
    X : np.ndarray, shape (784, m)
    parameters : dict

    Returns
    -------
    predictions : np.ndarray, shape (1, m)
        Predicted class index (0–9) for each example.

    Hint
    ----
    Use np.argmax on the softmax output (axis=0).
    """
    AL, _ = forward_propagation(X, parameters)
    predictions = np.argmax(AL, axis=0, keepdims=True)
    return predictions


def accuracy(predictions: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute classification accuracy.

    Parameters
    ----------
    predictions : np.ndarray, shape (1, m)
    Y : np.ndarray, shape (1, m)

    Returns
    -------
    acc : float
        Fraction of correct predictions (0.0 to 1.0).
    """
    acc = np.mean(predictions == Y)
    return float(acc)

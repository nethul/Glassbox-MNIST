import numpy as np
import matplotlib
matplotlib.use('Agg')
from utils import load_data, one_hot_encode, show_image
from network import (
    initialize_parameters,
    forward_propagation,
    compute_cost,
    backward_propagation,
    update_parameters,
    predict,
    accuracy,
)
from saliency import compute_saliency_map, render_heatmap



# ============================================================
# STEP 1 — TRAIN THE BASELINE CLASSIFIER
# ============================================================

def train(X_train: np.ndarray, Y_train: np.ndarray,
          layer_dims: list[int], learning_rate: float = 0.01,
          num_iterations: int = 1000, batch_size: int = 256,
          print_cost: bool = True) -> dict:
    """
    Train the L-layer network using mini-batch gradient descent.

    Parameters
    ----------
    X_train : shape (784, m)
    Y_train : shape (1, m)          — integer labels
    layer_dims : e.g. [784, 128, 64, 10]
    learning_rate : float
    num_iterations : int
    batch_size : int
        Size of each mini-batch.
    print_cost : bool

    Returns
    -------
    parameters : dict
        The trained weights and biases.
    """
    # 1. One-hot encode Y_train
    Y_one_hot = one_hot_encode(Y_train, num_classes=layer_dims[-1])
    
    # 2. Initialize parameters
    parameters = initialize_parameters(layer_dims)
    
    m = X_train.shape[1]
    
    # 3. Optimization loop
    for i in range(num_iterations):
        # Sample a random mini-batch
        shuffled_indices = np.random.choice(m, batch_size, replace=False)
        X_batch = X_train[:, shuffled_indices]
        Y_batch_one_hot = Y_one_hot[:, shuffled_indices]
        
        # a. Forward propagation
        AL, cache = forward_propagation(X_batch, parameters)
        
        # b. Compute cost
        cost = compute_cost(AL, Y_batch_one_hot)
        
        # c. Backward propagation
        grads = backward_propagation(AL, Y_batch_one_hot, parameters, cache)
        
        # d. Update parameters
        parameters = update_parameters(parameters, grads, learning_rate)
        
        # e. Print cost every 100 iterations
        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i:4d}: {cost:.6f}")
            
    return parameters


# ============================================================
# STEP 2 & 3 — GENERATE SALIENCY MAPS
# ============================================================

def explain(X_test: np.ndarray, Y_test: np.ndarray,
            parameters: dict, sample_indices: list[int]) -> None:
    """
    For each sample index, generate and display a saliency heatmap.

    Parameters
    ----------
    X_test : shape (784, m)
    Y_test : shape (1, m)
    parameters : dict
    sample_indices : list of int
        Which test examples to explain.

    Pseudocode
    ----------
    For each index i in sample_indices:
        1. Extract the single image:  x = X_test[:, i:i+1]
        2. Predict its class.
        3. Compute saliency map via compute_saliency_map(x, parameters).
        4. Render the heatmap with render_heatmap().
    """
    for idx in sample_indices:
        x = X_test[:, idx:idx+1]
        true_label = int(Y_test[0, idx])
        
        # 2. Predict its class
        preds = predict(x, parameters)
        predicted_label = int(preds[0, 0])
        
        # 3. Compute saliency map
        saliency_vector = compute_saliency_map(x, parameters, target_class=predicted_label)
        
        # 4. Render the heatmap
        save_path = f"saliency_sample_{idx}.png"
        render_heatmap(x, saliency_vector, predicted_label, true_label, save_path=save_path)
        print(f"Saved saliency map for sample {idx} (label {true_label}, pred {predicted_label}) to {save_path}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # --- Configuration ---
    TRAIN_CSV = "mnist_train.csv"
    TEST_CSV = "mnist_test.csv"
    LAYER_DIMS = [784, 128, 64, 10]   # Feel free to experiment
    LEARNING_RATE = 0.05
    NUM_ITERATIONS = 500

    # --- Load Data ---
    print("Loading data...")
    X_train, Y_train = load_data(TRAIN_CSV)
    X_test, Y_test   = load_data(TEST_CSV)
    print(f"Loaded {X_train.shape[1]} training samples and {X_test.shape[1]} test samples.")

    # --- Train ---
    print("Training the baseline classifier...")
    parameters = train(X_train, Y_train, LAYER_DIMS,
                       LEARNING_RATE, NUM_ITERATIONS)

    # --- Evaluate ---
    preds_test = predict(X_test, parameters)
    print(f"Test Accuracy: {accuracy(preds_test, Y_test) * 100:.2f}%")

    # --- Explain ---
    # Pick a few test images to generate saliency maps for
    print("Generating saliency heatmaps for test examples...")
    explain(X_test, Y_test, parameters, sample_indices=[0, 2, 11, 110])

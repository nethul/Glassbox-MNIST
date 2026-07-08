import numpy as np
import matplotlib.pyplot as plt
from network import forward_propagation, relu_derivative


def compute_saliency_map(X: np.ndarray, parameters: dict,
                         target_class: int | None = None) -> np.ndarray:
    """
    Compute a saliency map for a SINGLE input image.

    This is the "reverse backprop" trick:
    Instead of  ∂L/∂W  (for training),
    we compute  ∂S_c/∂X  (for explanation).

    Parameters
    ----------
    X : np.ndarray, shape (784, 1)
        A single flattened image.
    parameters : dict
        Trained network weights/biases.
    target_class : int or None
        The class to explain. If None, use the predicted class
        (i.e., the argmax of the softmax output).

    Returns
    -------
    saliency : np.ndarray, shape (784, 1)
        The gradient of the target class score with respect to
        each input pixel.  |dX| gives pixel importance.

    Steps
    -----
    1. FORWARD PASS
       Run the standard forward_propagation to get AL and cache.

    2. SET THE SEED GRADIENT
       Create dAL with the same shape as AL, filled with zeros.
       Set dAL[target_class, 0] = 1.
       (This says: "I only care about this one class score.")

    3. BACKWARD PASS (modified)
       Propagate dAL backward through every layer, exactly like
       training backprop, BUT:
         - At the output layer:  dZ[L] = dAL  (no subtraction of Y)
         - At hidden layers:     dA[l] = W[l+1]ᵀ · dZ[l+1]
                                 dZ[l] = dA[l] * relu_derivative(Z[l])

    4. COMPUTE dX
       At layer 1:  dX = W[1]ᵀ · dZ[1]
       This is your raw saliency vector.

    5. Return the absolute value: np.abs(dX)
    """
    # 1. FORWARD PASS
    AL, cache = forward_propagation(X, parameters)

    # 2. SET THE SEED GRADIENT
    if target_class is None:
        target_class = np.argmax(AL, axis=0)[0]
    
    dAL = np.zeros_like(AL)
    dAL[target_class, 0] = 1.0

    # 3. BACKWARD PASS (modified)
    L = len(parameters) // 2
    dZ_curr = dAL

    for l in range(L - 1, 0, -1):
        W_next = parameters["W" + str(l+1)]
        dA_l = np.dot(W_next.T, dZ_curr)
        Z_l = cache["Z" + str(l)]
        dZ_l = dA_l * relu_derivative(Z_l)
        dZ_curr = dZ_l

    # 4. COMPUTE dX
    W1 = parameters["W1"]
    dX = np.dot(W1.T, dZ_curr)

    # 5. Return absolute value
    return np.abs(dX)


def render_heatmap(image_vector: np.ndarray, saliency_vector: np.ndarray,
                   predicted_label: int, true_label: int | None = None,
                   save_path: str | None = None) -> None:
    """
    Overlay the saliency heatmap on top of the original image.

    Parameters
    ----------
    image_vector : np.ndarray, shape (784,) or (784, 1)
        Original pixel values (normalized [0,1]).
    saliency_vector : np.ndarray, shape (784,) or (784, 1)
        Absolute saliency values from compute_saliency_map().
    predicted_label : int
        The class the network predicted.
    true_label : int or None
        The ground-truth label (for the title).
    save_path : str or None
        Path to save the resulting heatmap image.
    """
    img = image_vector.reshape(28, 28)
    saliency = saliency_vector.reshape(28, 28)

    # Normalize saliency to [0, 1]
    max_val = np.max(saliency)
    if max_val > 0:
        saliency = saliency / max_val

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. Original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # 2. Saliency map alone
    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title("Saliency Map")
    axes[1].axis('off')

    # 3. Overlay
    axes[2].imshow(img, cmap='gray')
    axes[2].imshow(saliency, cmap='jet', alpha=0.5)
    axes[2].set_title("Overlay Heatmap")
    axes[2].axis('off')

    title_text = f"Predicted: {predicted_label}"
    if true_label is not None:
        title_text += f" (True: {true_label})"
    fig.suptitle(title_text, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close(fig)

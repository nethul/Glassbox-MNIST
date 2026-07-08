import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DATA LOADING
# ============================================================

def load_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST data from a CSV file.

    Each row in the CSV is: label, pixel_0, pixel_1, ..., pixel_783

    Returns
    -------
    X : np.ndarray, shape (784, m)
        Pixel values normalized to [0, 1] as float32.
        Each column is one flattened 28×28 image.
    Y : np.ndarray, shape (1, m)
        Integer labels (0–9).
    """
    database = pd.read_csv(csv_path)
    X = database.values[:, 1:].T
    Y = database.values[:, 0:1].T
    X = (X / 255.0).astype(np.float32)
    return X, Y


def one_hot_encode(Y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded matrix as float32.

    Parameters
    ----------
    Y : np.ndarray, shape (1, m)
        Integer class labels.
    num_classes : int
        Total number of classes (10 for MNIST digits).

    Returns
    -------
    Y_one_hot : np.ndarray, shape (num_classes, m)
        One-hot encoded labels.
        Column j has a 1 in row Y[0, j] and 0 elsewhere.
    """
    m = Y.size
    one_hot_Y = np.zeros((num_classes, m), dtype=np.float32)
    one_hot_Y[Y.flatten(), np.arange(m)] = 1.0
    return one_hot_Y


# ============================================================
# VISUALIZATION
# ============================================================

def show_image(pixel_vector: np.ndarray, label: int | None = None) -> None:
    """
    Display a single MNIST image.

    Parameters
    ----------
    pixel_vector : np.ndarray, shape (784,) or (784, 1)
        Flattened pixel values (already normalized to [0,1]).
    label : int or None
        Optional label to show as the plot title.
    """
    reshaped = pixel_vector.reshape(28, 28)
    plt.imshow(reshaped, cmap='gray')
    if label is not None:
        plt.title(str(label))
    plt.show()

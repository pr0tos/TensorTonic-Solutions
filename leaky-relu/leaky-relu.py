import numpy as np

def leaky_relu(x: list | float, alpha: float = 0.01) -> np.ndarray:
    """
    Returns elementwise Leaky ReLU values as a NumPy array matching the input shape.
    """
    for i in range(len(x)):
        if x[i]>=0:
            x[i] = x[i]
        else:
            x[i] = alpha*x[i]
    return np.array(x)
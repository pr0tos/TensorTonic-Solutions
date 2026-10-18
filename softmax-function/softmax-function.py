import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    x = np.array(x)
    try:
        p = []
        x.shape[1]
        for i in range(x.shape[0]):
            p.append(np.exp(x[i] - np.max(x[i])) / np.sum(np.exp(x[i] - np.max(x[i])))) 
        p = np.array(p)
    except IndexError:
        p = np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    return p
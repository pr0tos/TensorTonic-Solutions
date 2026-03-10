import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    ans = 1 / (1 + np.exp(-np.array(x)) + 0.000000000001)
    return ans
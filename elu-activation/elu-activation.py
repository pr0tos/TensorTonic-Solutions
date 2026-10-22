import math
import numpy as np
def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    x = np.array(x)
    return list(np.where(x < 0, alpha*(np.exp(x)-1), x))
import torch

def squared_error_loss_gradients(predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
    """
    Returns a scalar loss tensor and a prediction-shaped gradient tensor.
    """
    if predictions != []:
        loss = ((predictions - targets)**2).sum()
        g = 2*(predictions - targets)
    else:
        return (0.0, [])
    return (loss, g)

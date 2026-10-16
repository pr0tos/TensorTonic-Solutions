import torch

def neuron_forward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> tuple:
    
    i = 0
    a = 0
    while i<=len(inputs)-1:
        a += inputs[i]*weights[i]
        i+=1
    return tuple([a+bias, torch.tanh(a+bias)])
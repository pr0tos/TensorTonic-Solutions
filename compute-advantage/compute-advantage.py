import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    ans=[]
    for i in range(len(rewards)):
        G_t=0
        for k in range(i, len(rewards)):
            G_t=G_t+(gamma ** (k-i)) * rewards[k]
        ans.append(G_t - V[i])
    return ans
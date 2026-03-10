def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    ans=[]
    for i in range(len(rewards)):
        G_t=0
        for k in range(i, len(rewards)):
            G_t=G_t+(gamma ** (k-i)) * rewards[k]
        ans.append(G_t)
    return ans
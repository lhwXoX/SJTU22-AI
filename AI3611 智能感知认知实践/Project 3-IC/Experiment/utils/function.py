import numpy as np

def scheduled_sampling(mode, epoch):
    # linear
    epsilon = 0.1
    k_linear = 1.0
    c_linear = 0.05
    # exponential
    k_exp = 0.9
    # sigmoid
    k_sigmoid = 5.0
    if mode == 'linear':
        return max(epsilon, k_linear - c_linear * epoch)
    elif mode == 'exponential':
        return k_exp ** epoch
    elif mode == 'sigmoid':
        return k_exp / (k_exp + np.exp(epoch / k_sigmoid))
    else:
        return 1.0
    
    
import numpy as np


def running_mean(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def running_std(x, N):
    std = np.zeros_like(x)
    if len(x) >= N:
        std[N-1:] = np.var(np.lib.stride_tricks.sliding_window_view(x, N), axis=-1)
    return std

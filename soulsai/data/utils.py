import numpy as np


def int_to_onehot(n, n_max):
    assert 0 <= n < n_max
    onehot = np.zeros(n_max, dtype=np.float32)
    onehot[n] = 1
    return onehot

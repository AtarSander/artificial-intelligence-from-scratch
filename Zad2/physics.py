import numpy as np


def target_function(x, p=200, v=0, m=200, g=-0.09):
    m += np.sum(x)
    cost = -np.sum(x)
    a = 0
    for i in range(200):
        if x[i]:
            m -= 1
            a = a + 40/m + g
        else:
            a = a + g
        v += a
        p += v
        if 0 < p < 2 and np.abs(v) < 2:
            return cost + 2000
        elif p < 0:
            return cost - 1000
    return cost

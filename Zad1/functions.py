import numpy as np


def f1(x):
    """A mathematical function"""
    if abs(x) > 1000:
        return np.inf
    else:
        return np.array(1/4 * np.power(x, 4))


def gradient_f1(x):
    """A mathematical function gradient, returns inf if gradient explodes"""
    if abs(x) > 1000:
        return np.inf
    else:
        return np.array(np.power(x, 3))


def f2(x):
    """A mathematical function"""
    x1, x2 = x
    return np.array(1.5 - np.exp(-x1**2 - x2**2) - 0.5 * np.exp(-(x1 - 1)**2 - (x2 + 2)**2))


def gradient_f2(x0):
    """A mathematical function gradient"""
    x1, x2 = x0
    return np.array([2 * x1 * np.exp(-x1**2 - x2**2) + (x1 - 1) * np.exp(-(x1 - 1)**2 - (x2 + 2)**2),
                     2 * x2 * np.exp(-x1**2 - x2**2) + (x2 + 2) * np.exp(-(x1 - 1)**2 - (x2 + 2)**2)])

import numpy as np


def f1(x):
    return 1/4 * x**4


def gradient_f1(x):
    return x**3


def f2(x):
    x1, x2 = x
    return 1.5 - np.exp(-x1**2 - x2**2) - 0.5 * np.exp(-(x1 - 1)**2 - (x2 + 2)**2)


def gradient_f2(x0):
    x1, x2 = x0
    return np.array([2 * x1 * np.exp(-x1**2 - x2**2) + (x1 - 1) * np.exp(-(x1 - 1)**2 - (x2 + 2)**2),
                     2 * x2 * np.exp(-x1**2 - x2**2) + (x2 + 2) * np.exp(-(x1 - 1)**2 - (x2 + 2)**2)])

import numpy as np
from layer import Layer


class Model:
    def __init__(self):
        self.layers = []

    def initialize_parameters(self):
        pass

    def add_module(self, name, module):
        if not (hasattr(module, "__self__") and isinstance(module.__self__, self)):
            raise ValueError

    def forward_propagation(self, X):
        A_prev = X
        for layer in self.layers:
            A, cache = layer.forward(A_prev)


    def backward_propagation(self, x):
        pass

    def ReLU(self, Z):
        cache = Z
        return np.maximum(Z, 0), cache

    def SoftMax(self, x):
        sigma = 1/(1+np.exp(x))
        d_sigma = sigma * (1-sigma)
        return sigma, d_sigma

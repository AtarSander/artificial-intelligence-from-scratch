import numpy as np
from layer import Layer


class Model:
    def __init__(self):
        self.layers = []

    def add_module(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, X):
        A_prev = X
        for layer in self.layers:
            A, cache = layer.forward(A_prev)


    def backward_propagation(self, x):
        pass


import numpy as np


class Layer:
    def __init__(self, layer_dims, activation):
        self.weights, self.bias = self.initialize_parameters(layer_dims)
        self.activation = activation.lower()

    def initialize_parameters(self, dimensions):
        self.weights = np.random.normal(0, 2 / dimensions[1], dimensions)
        self.bias = np.zeros((dimensions[1],1))

    def forward(self, X):
        Z = self.forward_linear(X)
        Z = self.activation(Z)
        return Z

    def forward_linear(self, X):
        w, b = self.weights, self.bias
        Z = np.dot(np.transpose(X), w) + b
        return Z

    def activation(self, Z):
        if self.activation == "relu":
            Z = self.ReLU(Z)
        elif self.activation == "softmax":
            Z = self.SoftMax(Z)
        return Z

    def backward(self, d_prev, A_prev):
        dZ = d_prev * self.cache["activation"]
        dW = np.dot(dZ, np.transpose(A_prev))
        db = np.sum(dZ, axis=1, keepdims=True)

    def ReLU(self, Z):
        self.cache["activation"] = np.ones(Z.shape)
        return np.maximum(Z, 0)

    def SoftMax(self, x):
        shift_x = x-np.max(x)
        e_x = np.exp(shift_x)
        softmax = e_x / np.sum(e_x)
        self.cache["activation"] = softmax * (1-softmax)
        return softmax

    def update_patameters(self, grads_w, grads_b, learning_rate):
        self.weights = self.weights - learning_rate * grads_w
        self.bias = self.bias - learning_rate * grads_b






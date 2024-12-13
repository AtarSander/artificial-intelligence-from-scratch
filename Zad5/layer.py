import numpy as np


class Layer:
    def __init__(self, layer_dims, activation):
        self.initialize_parameters(layer_dims)
        self.activation_type = activation.lower()
        self.cache = {}

    def initialize_parameters(self, dimensions):
        self.weights = np.array(np.random.normal(0, 2 / dimensions[1], list(dimensions)[::-1]))
        self.bias = np.zeros((dimensions[1],1))

    def forward(self, X):
        self.cache["input"] = X
        Z = self.forward_linear(X)
        Z = self.activation(Z)
        return Z

    def forward_linear(self, X):
        w, b = self.weights, self.bias
        Z = np.dot(w, X) + b
        return Z

    def activation(self, Z):
        if self.activation_type == "relu":
            Z = self.relu(Z)
        elif self.activation_type == "softmax":
            Z = self.softmax(Z)
        return Z

    def backward(self, d_prev):
        dZ = d_prev * self.cache["activation"]
        dW = np.dot(dZ, np.transpose(self.cache["input"]))
        db = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(np.transpose(self.weights), dZ)
        return dW, db, dA_prev

    def relu(self, Z):
        self.cache["activation"] = np.ones(Z.shape)
        return np.maximum(Z, 0)

    def softmax(self, x):
        shift_x = x-np.max(x)
        e_x = np.exp(shift_x)
        softmax = e_x / np.sum(e_x)
        self.cache["activation"] = softmax * (1-softmax)
        return softmax

    def update_parameters(self, grads_w, grads_b, learning_rate):
        self.weights = self.weights - learning_rate * grads_w
        self.bias = self.bias - learning_rate * grads_b






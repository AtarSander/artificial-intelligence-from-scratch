import numpy as np
import math


class Model:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()

    def get_parameters(self):
        return self.parameters

    def initialize_parameters(self):
        np.random.seed(2)
        parameters = {}
        L = len(self.layer_dims)
        for layer_ind in range(1, L):
            parameters["W" + str(layer_ind)] = np.random.randn(
                self.layer_dims[layer_ind], self.layer_dims[layer_ind - 1]
            ) * np.sqrt(2 / self.layer_dims[layer_ind - 1])
            parameters["b" + str(layer_ind)] = np.zeros((self.layer_dims[layer_ind], 1))
        return parameters

    def relu(self, Z):
        cache = Z
        A = np.maximum(0, Z)
        return A, cache

    def softmax(self, Z):
        cache = Z
        e_x = np.exp(Z - np.max(Z))
        A = e_x / e_x.sum(axis=0, keepdims=True)
        return A, cache

    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, prev_A, W, b, activation_function):
        Z, linear_cache = self.linear_forward(prev_A, W, b)
        if activation_function == "relu":
            A, activation_cache = self.relu(Z)

        elif activation_function == "softmax":
            A, activation_cache = self.softmax(Z)
        cache = (linear_cache, activation_cache)
        return A, cache

    def forward_propagation(self, X):
        caches = []
        L = len(self.parameters) // 2
        A = X
        for layer_ind in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(
                A_prev,
                self.parameters["W" + str(layer_ind)],
                self.parameters["b" + str(layer_ind)],
                "relu",
            )
            caches.append(cache)

        AL, cache = self.linear_activation_forward(
            A, self.parameters["W" + str(L)], self.parameters["b" + str(L)], "softmax"
        )
        caches.append(cache)
        return AL, caches

    def cost_function(self, AL, Y):
        m = Y.shape[1]
        epsilon = 1e-8
        cost = -1 / m * np.sum(np.multiply(Y, np.log(AL + epsilon)))
        cost = np.squeeze(cost)
        return cost

    def relu_backward(self, dA, activ_cache):
        Z = activ_cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def softmax_backward(self, dA, activ_cache):
        Z = activ_cache
        e_x = np.exp(Z - np.max(Z))
        A = e_x / e_x.sum(axis=0)
        dZ = dA * A * (1 - A)
        return dZ

    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation_function):
        linear_cache, activ_cache = cache
        if activation_function == "relu":
            dZ = self.relu_backward(dA, activ_cache)
        elif activation_function == "softmax":
            dZ = self.softmax_backward(dA, activ_cache)

        dA, dW, db = self.linear_backward(dZ, linear_cache)
        return dA, dW, db

    def backward_propagation(self, AL, Y, caches):
        gradients = {}
        L = len(caches)
        dAL = AL - Y
        current_cache = caches[L - 1]
        dA_prev, dW, db = self.linear_activation_backward(dAL, current_cache, "softmax")
        gradients["dA" + str(L - 1)] = dA_prev
        gradients["dW" + str(L)] = dW
        gradients["db" + str(L)] = db

        for layer_ind in reversed(range(L - 1)):
            current_cache = caches[layer_ind]
            dA_prev, dW, db = self.linear_activation_backward(
                gradients["dA" + str(layer_ind + 1)], current_cache, "relu"
            )
            gradients["dA" + str(layer_ind)] = dA_prev
            gradients["dW" + str(layer_ind + 1)] = dW
            gradients["db" + str(layer_ind + 1)] = db
        return gradients

    def update_parameters(self, gradients, learning_rate):
        L = len(self.parameters) // 2
        parameters = self.parameters.copy()
        for layer_ind in range(L):
            parameters["W" + str(layer_ind + 1)] = (
                parameters["W" + str(layer_ind + 1)]
                - learning_rate * gradients["dW" + str(layer_ind + 1)]
            )
            parameters["b" + str(layer_ind + 1)] = (
                parameters["b" + str(layer_ind + 1)]
                - learning_rate * gradients["db" + str(layer_ind + 1)]
            )
        return parameters

    def create_batches(self, X, Y, batch_size, seed=42):
        np.random.seed(seed)
        m = X.shape[1]
        batches = []
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((10, m))
        complete_batches_num = math.floor(m / batch_size)
        for i in range(complete_batches_num):
            batch_X = shuffled_X[:, i * batch_size : (i + 1) * batch_size]
            batch_Y = shuffled_Y[:, i * batch_size : (i + 1) * batch_size]
            batches.append((batch_X, batch_Y))

        if m % batch_size != 0:
            batch_X = shuffled_X[:, int(m / batch_size) * batch_size :]
            batch_Y = shuffled_Y[:, int(m / batch_size) * batch_size :]
            batches.append((batch_X, batch_Y))

        return batches

    def train(self, X, Y, learning_rate, epochs, batch_size, seed, print_cost):
        costs = []
        for i in range(epochs):
            cost_total = 0
            seed = seed + 1
            batches = self.create_batches(X, Y, batch_size, seed)
            for batch in batches:
                (batch_X, batch_Y) = batch
                AL, caches = self.forward_propagation(batch_X)
                cost = self.cost_function(AL, batch_Y)
                cost_total += cost
                gradients = self.backward_propagation(AL, batch_Y, caches)
                self.parameters = self.update_parameters(gradients, learning_rate)
            cost_avg = cost_total / len(batches)
            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost_avg}\n")
                costs.append(cost_avg)

        return costs

    def predict(self, X, Y):
        AL, _ = self.forward_propagation(X)
        predictions = np.argmax(AL, axis=0)
        Y = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == Y)
        return accuracy

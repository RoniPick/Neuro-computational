import numpy as np
import random


class Adaline:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.errors = []
        self.weights = []
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.random.rand(n_features + 1)
        self.errors = []
        for epoch in range(self.epochs):
            activation = self.activation(np.dot(X, self.weights[1:]) + self.weights[0])
            errors = y - activation
            self.weights[0] += self.learning_rate * errors.sum()
            self.weights[1:] += self.learning_rate * np.dot(X.T, errors)
            self.errors.append(errors.sum())

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(np.dot(X, self.weights[1:]) + self.weights[0]) >= 0, 1, -1)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        return correct / len(y)

    # def fit(self, X, y, tolerance=1e-4):
    #     n_features = X.shape[1]
    #     self.weights = np.random.rand(n_features + 1).astype(float)
    #     prev_error = np.inf
    #
    #     for epoch in range(self.epochs):
    #         error_sum = 0.0
    #         for i in range(len(X)):
    #             activation = self.activation(
    #                 np.dot(self.weights[1:].astype(float), X[i]).astype(float) + self.weights[0].astype(float))
    #             errors = y[i] - activation
    #             self.weights[0] += self.learning_rate * errors
    #             self.weights[1:] += self.learning_rate * X[i] * errors
    #             error_sum += errors ** 2
    #
    #         self.errors.append(error_sum)
    #
    #         if abs(prev_error - error_sum) < tolerance:
    #             break
    #
    #         prev_error = error_sum
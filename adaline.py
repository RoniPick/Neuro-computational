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
        # self.errors = []
        for epoch in range(self.epochs):
            error_sum = 0.0
            for i in range(len(X)):
                activation = self.activation(np.dot(self.weights[1:], X) + self.weights[0])
                errors = y - activation
                self.weights[0] += self.learning_rate * errors.sum()
                self.weights[1:] += self.learning_rate * np.dot(X.T, errors)
                error_sum += errors ** 2
            self.errors.append(error_sum)
            if error_sum == 0:
                break

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(np.dot(self.weights[1:], X) + self.weights[0]) >= 0, 1, -1)

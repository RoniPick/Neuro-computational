import numpy as np


class Adaline:
    def __init__(self, learning_rate=0.001, epochs=600):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.cost = None

    def fit(self, X, y):
        # initialize the weights
        self.w = np.zeros(X.shape[1] + 1)  # +1 for the bias term
        self.cost = []

        # train the model
        for epoch in range(self.epochs):
            y_pred = self.predict(X)  # make predictions
            error = y - y_pred  # compute the error
            self.w[1:] += self.learning_rate * X.T.dot(error)  # update the weights
            self.w[0] += self.learning_rate * error.sum()  # update the bias term

            # calculating cost
            cost = (error ** 2).sum() / 2.0
            self.cost.append(cost)
        return self

    def predict(self, X):
        z = X.dot(self.w[1:]) + self.w[0]  # compute the net input
        y_pred = np.where(z >= 0, 1, -1)  # apply the activation function
        return y_pred
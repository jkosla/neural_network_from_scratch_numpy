import numpy as np


class GradientDescent:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, layer, dW, db):
        layer.weights -= self.lr * dW
        layer.bias -= self.lr * db.T


class LossFunction:
    @staticmethod
    def loss(y_true, y_pred):
        return np.square(y_true - y_pred).mean()

    @staticmethod
    def gradient(y_true, y_pred):
        return -(y_true - y_pred)


class LinearFunction:
    def __init__(self, row_size, column_size):
        self.weights = np.random.randn(row_size, column_size)
        self.bias = np.random.randn(row_size, 1)


class SimpleClassifier:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.linear1 = LinearFunction(num_hidden, num_inputs)
        self.act_fn = np.tanh
        self.linear2 = LinearFunction(num_outputs, num_hidden)

    def act_fn_deriv(self, h):
        return 1 - np.tanh(h) ** 2

    def forward(self, x):
        h = x @ self.linear1.weights.T + self.linear1.bias.T
        a = self.act_fn(h)
        y_pred = a @ self.linear2.weights.T + self.linear2.bias.T
        self.a, self.h = a, h
        return y_pred

    def backprop(self, x, y_true, y_pred):
        upstream_gradient = LossFunction.gradient(y_true, y_pred)

        dL_dy = upstream_gradient[:, np.newaxis]
        dy_da = self.linear2.weights
        da_dh = self.act_fn_deriv(self.h)
        dh_dW1 = x
        delta1 = (dL_dy @ dy_da) * da_dh

        dL_dW1 = delta1.T @ dh_dW1
        dL_db1 = delta1.sum(axis=0, keepdims=True)
        dL_dW2 = dL_dy.T @ self.a
        dL_db2 = upstream_gradient.sum(axis=-1, keepdims=True)

        return dL_dW1, dL_db1, dL_dW2, dL_db2

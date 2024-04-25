import numpy as np


class NaiveBayes:
    def __init__(self, X, y, cx, cy):
        self.X, self.y = X, y
        self.cx, self.cy = cx, cy
        self.phi_x = np.zeros((self.X.shape[1], self.cy, self.cx))
        self.phi_y = np.zeros(self.cy)

    def train(self):
        M, N = self.X.shape

        for i in range(M):
            for j in range(N):
                self.phi_x[j, self.y[i], self.X[i, j]] += 1
            self.phi_y[self.y[i]] += 1

        for c in range(self.cy):
            self.phi_x[:, c, :] /= self.phi_y[c]
        self.phi_y /= M

    def predict(self, X):
        M, N = X.shape
        p_xy = np.ones((M, self.cy))
        for i in range(M):
            for c in range(self.cy):
                for j in range(N):
                    p_xy[i, c] *= self.phi_x[j, c, X[i, j]]
                p_xy[i, c] *= self.phi_y[c]
        return p_xy.argmax(axis=1)


class NaiveBayesLaplace(NaiveBayes):
    def train(self):
        M, N = self.X.shape

        for i in range(M):
            for j in range(N):
                self.phi_x[j, self.y[i], self.X[i, j]] += 1
            self.phi_y[self.y[i]] += 1
        self.phi_x += 1

        for c in range(self.cy):
            self.phi_x[:, c, :] /= (self.cy + self.phi_y[c])
        self.phi_y = (1 + self.phi_y) / (M + self.cy)


class LogisticRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X, theta):
        return (1+np.exp(-np.dot(X, theta)))**-1

    def train(self, num_epochs, step_size=0.1, batch_size=100):
        N, D = self.X.shape
        theta = np.zeros([D, 1])
        loss = 0
        thetas = []
        for epoch in range(num_epochs):
            gradient = self._grad(self.X, self.y, theta)
            theta = self._step(theta, gradient, step_size)
            ypred = self.predict(self.X, theta)
            loss = self.loss(self.y, ypred)
            thetas += [theta]

        return thetas, loss

    def loss(self, y, ypred):
        return np.sum(np.where(y != np.where(ypred < 0.5, 0, 1), 1, 0))/y.shape[0]

    def _grad(self, X, y, theta):
        return np.dot(X.T, self.predict(X, theta) - y) / y.shape[0]

    def _step(self, theta, grad, step_size):
        return theta - step_size * grad
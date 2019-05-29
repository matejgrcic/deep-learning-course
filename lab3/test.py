import unittest
import numpy as np
from RNN import *
from dataset import *


def numerical_gradient(x, lf, eps=1e-8):
    grad = np.zeros_like(x)

    for i in np.ndindex(*x.shape):
        x0 = np.copy(x)
        x0[i] += eps
        up = lf(x0)
        x0 = np.copy(x)
        x0[i] -= eps
        down = lf(x0)
        grad[i] = (up - down) / (2 * eps)

    return grad


class Test(unittest.TestCase):

    def setUp(self):

        self.B = 4
        self.H = 3
        self.T = 20
        self.V = 2
        self.rnn = RNN(self.H, self.T, self.V, 0.1)
        x = np.ones(shape=(self.B, 2))
        self.x = x

        self.h0 = np.array([[1, 2, 3], [4, 5, 6], [7, 6, 8], [1, 2, 3]])*0.1



    def test_dh(self):
        self.rnn = RNN(self.H, self.T, self.V, 0.1)
        h, cache = self.rnn.rnn_step_forward(self.x, self.h0)
        grad, dU, dW, db = self.rnn.rnn_step_backward(np.ones([self.B, self.rnn.hidden_size]), cache)
        grad = np.average(grad, axis=0)

        def lf(h0):
            h, _ = self.rnn.rnn_step_forward(self.x, h0)
            return np.average(np.sum(h, axis=1))

        num_grad = numerical_gradient(self.h0, lf)
        num_grad = np.sum(num_grad, axis=0)

        np.testing.assert_allclose(
            grad * self.B,
            num_grad,
            rtol=1e-4
        )

    def test_dW(self):
        self.rnn = RNN(self.H, self.T, self.V, 0.1)
        h, cache = self.rnn.rnn_step_forward(self.x, self.h0)
        grad, dU, dW, db = self.rnn.rnn_step_backward(np.ones([self.B, self.rnn.hidden_size]), cache)

        def lf(w):
            temp = self.rnn.W
            self.rnn.W = w
            h, _ = self.rnn.rnn_step_forward(self.x, self.h0)
            self.rnn.W = temp
            return np.average(np.sum(h, axis=1))

        num_grad = numerical_gradient(self.rnn.W, lf)

        np.testing.assert_allclose(
            dW,
            num_grad,
            rtol=1e-4
        )

    def test_dU(self):
        self.rnn = RNN(self.H, self.T, self.V, 0.1)
        
        h, cache = self.rnn.rnn_step_forward(self.x, self.h0)
        grad, dU, dW, db = self.rnn.rnn_step_backward(np.ones([self.B, self.rnn.hidden_size]), cache)

        def fn(u):
            temp = self.rnn.U
            self.rnn.U = u
            h, _ = self.rnn.rnn_step_forward(self.x, self.h0)
            self.rnn.U = temp
            return np.average(np.sum(h, axis=1))

        np.testing.assert_allclose(
            dU,
            numerical_gradient(self.rnn.U, fn),
            rtol=1e-4
        )

    def test_db(self):
        self.rnn = RNN(self.H, self.T, self.V, 0.1)
        
        h, cache = self.rnn.rnn_step_forward(self.x, self.h0)
        grad, dU, dW, db = self.rnn.rnn_step_backward(np.ones([self.B, self.rnn.hidden_size]), cache)

        def fn(b):
            temp = self.rnn.b
            self.rnn.b = b
            h, _ = self.rnn.rnn_step_forward(self.x, self.h0)
            self.rnn.b = temp
            return np.average(np.sum(h, axis=1))

        np.testing.assert_allclose(
            db,
            numerical_gradient(self.rnn.b, fn),
            1e-4
        )




if __name__ == '__main__':
    unittest.main()
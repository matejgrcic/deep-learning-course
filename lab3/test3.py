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
        H = 3
        T = 2
        V = 2
        self.rnn = RNN(H, T, V, 0.1)
        x = np.zeros(shape=(2, 2, 2))
        x[0, :, 1] = 1
        x[1, :, 0] = 1
        self.x = x

        self.h0 = np.array([[1, 2, 3], [4, 5, 6]])*0.1
        self.rnn.U = np.array([[1, 2, 3], [4, 5, 6]])*0.1
        self.rnn.W = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])*0.1
        self.rnn.b = np.array([1, 2, 3])*(-0.1)

        self.rnn.V = np.array([[1, 2], [4, 5], [7, 8]]).T*0.2
        self.rnn.c = np.array([1, 2])*(-0.2)


    def test_1(self):
        h, _ = self.rnn.rnn_forward(self.x, self.h0)
        A = np.array([[[0.53704957, 0.57836341, 0.6169093],
                       [0.76859548, 0.83061579, 0.87716807]],
                      [[0.57836341, 0.66959026, 0.74427687],
                       [0.68932383, 0.78020306, 0.84690738]]
                     ])
        np.testing.assert_allclose(h, A)

    def test_2(self):
        h, _ = self.rnn.rnn_forward(self.x, self.h0)
        o = self.rnn.output(h)
        yhat = softmax(o, axis=2)
        A = np.array([[[0.4634492, 0.5365508],
                       [0.42671274, 0.57328726]],
                      [[0.45055065, 0.54944935],
                       [0.4345559, 0.5654441]]
                      ])
        np.testing.assert_allclose(yhat, A)


    def test_3(self):
        h, _ = self.rnn.rnn_forward(self.x, self.h0)
        o = self.rnn.output(h)
        yhat = softmax(o, axis=2)
        y = np.array([[[0, 1],
                       [1, 0]],
                      [[1, 0],
                       [1, 0]]
                      ])

        loss = - np.sum(y * np.log(yhat)) / h.shape[0]
        np.testing.assert_allclose(loss, 1.5524768732468268)

    def test_5(self):
        h, _ = self.rnn.rnn_forward(self.x, self.h0)
        y = np.array([[[0, 1],
                       [1, 0]],
                      [[1, 0],
                       [1, 0]]
                      ])
        loss, _, _, _ = self.rnn.output_loss_and_grads(h, y)
        np.testing.assert_allclose(loss, 1.5524768732468268)


    def test_4(self):
        h, _ = self.rnn.rnn_forward(self.x, self.h0)
        y = np.array([[[0, 1],
                       [1, 0]],
                      [[1, 0],
                       [1, 0]]
                      ])
        loss, _, _, _ = self.rnn.output_loss_and_grads(h, y)
        np.testing.assert_allclose(loss, 1.5524768732468268)



if __name__ == '__main__':
    unittest.main()
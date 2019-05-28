from RNN import RNN
import numpy as np
np.random.seed(69)
# def numerical_gradient(x, lf, eps=1e-8):
#     grad = np.zeros_like(x)

#     for i in np.ndindex(*x.shape):
#         x0 = np.copy(x)
#         x0[i] += eps
#         up = lf(x0)
#         x0 = np.copy(x)
#         x0[i] -= eps
#         down = lf(x0)
#         grad[i] = (up - down) / (2 * eps)

#     return grad
def numerical_gradient(var, fn, eps=1e-7):
    grad = np.zeros_like(var)
    for idx in np.ndindex(*var.shape):
        init = np.copy(var)
        init[idx] += eps
        up = fn(init)
        init = np.copy(var)
        init[idx] -= eps
        down = fn(init)
        grad[idx] = (up - down) / (2 * eps)
        # print(idx, up - down)

    return grad

N = 2
H = 5
T = 3
V = 7
x = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])
h_prev = np.ones((N, H))
grad_next = np.ones((N, H))

x_long = np.array([[[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14], [1, 2, 3, 4, 5, 6, 7]],
                   [[7, 6, 5, 4, 3, 2, 1], [8, 9, 10, 11, 12, 13, 14], [14, 13, 12, 11, 10, 9, 8]]])

y_long = np.array([[[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]],
                   [[0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]]])


def test_backprob_W():
    rnn = RNN(H, T, V, 0.01)
    h, cache = rnn.rnn_step_forward(x, h_prev)
    _, _, dW, _ = rnn.rnn_step_backward(grad_next, cache)

    def fn(w):
        rnn.W = w
        h, _ = rnn.rnn_step_forward(x, h_prev)  # Fictional loss = sum(h)
        return np.average(np.sum(h, axis=1))
    np.testing.assert_allclose(
        dW,
        numerical_gradient(rnn.W, fn),
        atol=1e-3
    )

def test_backprob_h():
    rnn = RNN(H, T, V, 0.01)
    h, cache = rnn.rnn_step_forward(x, h_prev)
    dh_prev, _, _, _ = rnn.rnn_step_backward(grad_next, cache)

    def fn(h0):
        h, _ = rnn.rnn_step_forward(x, h0)
        return np.average(np.sum(h, axis=1))  # Fictional loss = sum(h)


    np.testing.assert_allclose(
        dh_prev,  # Sum of all gradients
        numerical_gradient(h, fn),
        atol=1e-3
    )

def test_backprob_U():
    rnn = RNN(H, T, V, 0.01)
    h, cache = rnn.rnn_step_forward(x, h_prev)
    _, dU, _, _ = rnn.rnn_step_backward(grad_next, cache)

    def fn(u):
        rnn.U = u
        h, _ = rnn.rnn_step_forward(x, h_prev)  # Fictional loss = sum(h)
        return np.average(np.sum(h, axis=1))

    np.testing.assert_allclose(
        dU,  # Average of all gradients
        numerical_gradient(rnn.U, fn),
        rtol=1e-3
    )

def test_backprob_b():
    rnn = RNN(H, T, V, 0.01)
    h, cache = rnn.rnn_step_forward(x, h_prev)
    _, _, _, db = rnn.rnn_step_backward(grad_next, cache)

    def fn(b):
        rnn.b = b
        h, _ = rnn.rnn_step_forward(x, h_prev)  # Fictional loss = sum(h)
        return np.average(np.sum(h, axis=1))

    np.testing.assert_allclose(
        db,  # Average of all gradients
        numerical_gradient(rnn.b, fn),
        rtol=1e-3
    )

def test_backprop_len_U():
    rnn = RNN(H, T, V, 0.01)
    h, cache = rnn.rnn_forward(x_long, h_prev)
    dh = np.ones((N, T, H))
    dU, dW, db = rnn.rnn_backward(dh, cache)

    def fn(u):
        rnn.U = u
        h, cache = rnn.rnn_forward(x_long, h_prev)
        sol = 0
        h = h.transpose(1, 0, 2)  # switch to time major
        for i, hh in enumerate(h):
            sol += np.average(np.sum((i + 1) * hh, axis=1))
        return sol

    np.testing.assert_allclose(
        dU,
        numerical_gradient(rnn.U, fn),
        rtol=1e-3
    )

def test_backprop_len_W():
    rnn = RNN(H, T, V, 0.01)
    h, cache = rnn.rnn_forward(x_long, h_prev)
    dh = np.ones((N, T, H))
    dU, dW, db = rnn.rnn_backward(dh, cache)

    def fn(w):
        rnn.W = w
        h, cache = rnn.rnn_forward(x_long, h_prev)
        sol = 0
        h = h.transpose(1, 0, 2)  # switch to time major
        for i, hh in enumerate(h):
            sol += np.average(np.sum((i + 1) * hh, axis=1))
        return sol

    np.testing.assert_allclose(
        dW,
        numerical_gradient(rnn.W, fn),
        rtol=1e-3
    )

def test_backprop_len_b():
    rnn = RNN(H, T, V, 0.01)
    h, cache = rnn.rnn_forward(x_long, h_prev)
    dh = np.ones((N, T, H))
    dU, dW, db = rnn.rnn_backward(dh, cache)

    def fn(b):
        rnn.b = b
        h, cache = rnn.rnn_forward(x_long, h_prev)
        sol = 0
        h = h.transpose(1, 0, 2)  # switch to time major
        for i, hh in enumerate(h):
            sol += np.average(np.sum((i + 1) * hh, axis=1))
        return sol

    np.testing.assert_allclose(
        db,
        numerical_gradient(rnn.b, fn),
        rtol=1e-3
    )

def test_output_dv():
    rnn = RNN(H, T, V, 0.01)
    h, cache = rnn.rnn_forward(x_long, h_prev)
    loss, dh, dV, dc = rnn.output_loss_and_grads(h, y_long)

    def fn(v):
        rnn.V = v
        loss, dh, dV, dc = rnn.output_loss_and_grads(h, y_long)
        return loss

    np.testing.assert_allclose(
        dV.T,
        numerical_gradient(rnn.V, fn),
        rtol=1e-3
    )

def test_output_dc():
    rnn = RNN(H, T, V, 0.01)
    h, cache = rnn.rnn_forward(x_long, h_prev)
    loss, dh, dV, dc = rnn.output_loss_and_grads(h, y_long)

    def fn(c):
        rnn.c = c
        loss, _, _, _ = rnn.output_loss_and_grads(h, y_long)
        return loss

    np.testing.assert_allclose(
        dc,  # Sum of all gradients
        numerical_gradient(rnn.c, fn),
        rtol=1e-3
    )


test_backprob_W()
test_backprob_h()
test_backprob_U()
test_backprob_b()
test_backprop_len_U()
test_backprop_len_W()
test_backprop_len_b()
test_output_dv()
test_output_dc()
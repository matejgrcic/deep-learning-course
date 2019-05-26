import numpy as np
from dataset import Data
from scipy.special import softmax
class RNN:

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.random.normal(0, 1e-2, (hidden_size, vocab_size))    # input projection
        self.W = np.random.normal(0, 1e-2, (hidden_size, hidden_size))   # hidden-to-hidden projection
        self.b = np.zeros((hidden_size))                                # input bias

        self.V = np.random.normal(0, 1e-2, (vocab_size, hidden_size))    # output projection
        self.c = np.zeros((vocab_size))                                 # output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U = np.zeros_like(self.U)
        self.memory_W = np.zeros_like(self.W)
        self.memory_V = np.zeros_like(self.V)
        self.memory_b = np.zeros_like(self.b)
        self.memory_c = np.zeros_like(self.c)

    def rnn_step_forward(self, x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        h_current = np.tanh(np.matmul(h_prev, W.T) + np.matmul(x, U.T) + b)
        cache = (h_prev, x, h_current)
        
        # return the new hidden state and a tuple of values needed for the backward step

        return h_current, cache

    def rnn_forward(self, x, h0):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        
        h = np.zeros((x.shape[0], self.sequence_length, self.hidden_size))
        h_current = h0
        batch_cache = []
        for i in range(self.sequence_length):
            h_current, cache = self.rnn_step_forward(x[:, i, :], h_current, self.U, self.W, self.b)
            h[:, i, :] = h_current
            batch_cache.append(cache)
        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step

        return h, batch_cache

    def rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass
        h_prev, x, h_current = cache
        # TODO is this element-wise
        da = grad_next * (np.ones(h_current.shape) - (h_current ** 2))
        dU = np.matmul(da.T, x)
        dW = np.matmul(da.T, h_prev)
        dh_prev = np.matmul(da, self.W)
        db = np.sum(da, axis=0)
        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters

        return dh_prev, dU, dW, db


    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity
        dU_total = np.zeros(self.U.shape)
        dW_total = np.zeros(self.W.shape)
        db_total = np.zeros(self.b.shape)
        for i in reversed(range(self.sequence_length)):
            _, dU, dW, db = self.rnn_step_backward(dh[:, i, :], cache[i])
            dU_total += dU
            dW_total += dW
            db_total += db
        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        return dU_total, dW_total, db_total


    def output(self, h):
        # Calculate the output probabilities of the network
        out = np.zeros((h.shape[0], self.sequence_length, self.vocab_size))
        for i in range(self.sequence_length):
            out[:, i, :] = np.matmul(h[:, i, :], self.V.T) + self.c
        return out

    def output_loss_and_grads(self, h, y):
        # Calculate the loss of the network for each of the outputs
        
        # h - hidden states of the network for each timestep. 
        #     the dimensionality of h is (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a tensor of dimension 
        #     batch_size x sequence_length x vocabulary size - you need to do this conversion prior to
        #     passing the argument. A fast way to create a one-hot vector from
        #     an id could be something like the following code:

        #   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
        #   y[batch_id][timestep][batch_y[timestep]] = 1

        #     where y might be a list or a dictionary.

        out = self.output(h)
        y_hat = softmax(out, axis=2)
        loss = - np.sum(y * np.log(y_hat))
        dout = (y_hat - y).T
        
        dV = np.zeros((self.hidden_size, self.vocab_size))
        dh = np.zeros(h.shape)
        dc = np.zeros((self.vocab_size))
        dh_future = np.zeros((h.shape[0], self.hidden_size))

        for i in reversed(range(self.sequence_length)):
            dV = np.matmul(dout[:, i, :], h[:, i, :]).T
            dc += np.sum(dout[:, i, :], axis=1)
            dh[:, i, :] += np.matmul(dout[:, i, :].T, self.V) + dh_future
            dh_future += dh[:, i, :]

        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h

        return loss, dh, dV / self.sequence_length, dc / self.sequence_length

    def update(self, dU, dW, db, dV, dc):
        self.memory_U += dU ** 2
        self.memory_W += dW ** 2
        self.memory_b += db ** 2
        self.memory_V += dV.T ** 2
        self.memory_c += dc ** 2

        delta = 1e-7
        self.U -= (self.learning_rate / (self.memory_U + delta)) * dU
        self.W -= (self.learning_rate / (self.memory_W + delta)) * dW
        self.b -= (self.learning_rate / (self.memory_b + delta)) * db
        self.V -= (self.learning_rate / (self.memory_V + delta)) * dV.T
        self.c -= (self.learning_rate / (self.memory_c + delta)) * dc
        # update memory matrices
        # perform the Adagrad update of parameters

    def step(self, h0, batch_x_oh, batch_y_oh):
        h, batch_cache = self.rnn_forward(batch_x_oh, h0)
        loss, dh, dV, dc = self.output_loss_and_grads(h, batch_y_oh)
        dU, dW, db = self.rnn_backward(dh, batch_cache)
        self.update(
            np.clip(dU, -5, 5),
            np.clip(dW, -5, 5),
            np.clip(db, -5, 5),
            np.clip(dV, -5, 5),
            np.clip(dc, -5, 5))
        return loss, h[:, -1, :]

if __name__ == "__main__":
    d = Data("./lab3/dataset/cornell movie-dialogs corpus/selected_conversations.txt", 10, 30)
    d.preprocess()
    d.create_minibatches()
    _, batch_x, batch_y = d.next_minibatch()
    batch_x_oh = (np.arange(70) == batch_x[...,None]).astype(int)
    rnn = RNN(100, 30, 70, 1e-3)

    h, batch_cache = rnn.rnn_forward(batch_x_oh, np.zeros((10, 100)))
    rnn.output_loss_and_grads(h, batch_y)
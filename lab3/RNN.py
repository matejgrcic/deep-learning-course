import numpy as np
from dataset import Data
from scipy.special import softmax
class RNN:

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.random.normal(0, 1e-2, (vocab_size, hidden_size))    # input projection
        self.W = np.random.normal(0, 1e-2, (hidden_size, hidden_size))   # hidden-to-hidden projection
        self.b = np.zeros((hidden_size))                                # input bias

        self.V = np.random.normal(0, 1e-2, (hidden_size, vocab_size))    # output projection
        self.c = np.zeros((vocab_size))                                 # output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U = np.zeros_like(self.U)
        self.memory_W = np.zeros_like(self.W)
        self.memory_V = np.zeros_like(self.V)
        self.memory_b = np.zeros_like(self.b)
        self.memory_c = np.zeros_like(self.c)

    def rnn_step_forward(self, x, h_prev):
        # A single time step forward of a recurrent neural network with a 
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        h_current = np.tanh(np.matmul(h_prev, self.W) + np.matmul(x, self.U) + self.b)
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
            h_current, cache = self.rnn_step_forward(x[:, i, :], h_current)
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
        da = grad_next * (np.ones(h_current.shape) - (h_current ** 2))
        dU = np.matmul(x.T, da)
        dW = np.matmul(h_prev.T, da)
        dh_prev = np.matmul(da, self.W.T)
        db = np.sum(da, axis=0)
        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters
        N = h_prev.shape[0]
        return dh_prev / N , dU / N, dW / N, db / N


    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a 
        # hyperbolic tangent nonlinearity
        dU_total = np.zeros(self.U.shape)
        dW_total = np.zeros(self.W.shape)
        db_total = np.zeros(self.b.shape)
        dh_previous = np.zeros((dh.shape[0], self.hidden_size))
        N = dh.shape[0]
        for i in reversed(range(self.sequence_length)):
            dh_previous, dU, dW, db = self.rnn_step_backward(dh[:, i, :] + dh_previous * N, cache[i])
            dU_total += dU
            dW_total += dW
            db_total += db
        # # compute and return gradients with respect to each parameter
        # # for the whole time series.
        # # Why are we not computing the gradient with respect to inputs (x)?

        return dU_total * N, dW_total * N, db_total * N

    def output(self, h):
        # Calculate the output probabilities of the network
        out = np.zeros((h.shape[0], self.sequence_length, self.vocab_size))
        for i in range(self.sequence_length):
            # out[:, i, :] = np.matmul(h[:, i, :], self.V.T) + self.c
            out[:, i, :] = self.output_step(h[:, i, :])
        return out

    def output_step(self, h):
        return np.matmul(h, self.V) + self.c

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
        N = h.shape[0]
        loss = - np.sum(y * np.log(y_hat)) / N
        dout = (y_hat - y) / N
        
        dV = np.zeros((self.hidden_size, self.vocab_size))
        dh = np.zeros(h.shape)
        dc = np.zeros((self.vocab_size))

        for i in range(self.sequence_length):
            dV += np.matmul(h[:, i, :].T, dout[:, i, :])
            dc += np.sum(dout[:, i, :].T, axis=1)
            dh[:, i, :] += np.matmul(dout[:, i, :], self.V.T)

        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h

        return loss, dh, dV, dc

    def update(self, dU, dW, db, dV, dc):
        self.memory_U += dU ** 2
        self.memory_W += dW ** 2
        self.memory_b += db ** 2
        self.memory_V += dV ** 2
        self.memory_c += dc ** 2

        delta = 1e-7
        self.U -= (self.learning_rate * dU / (np.sqrt(self.memory_U + delta)))
        self.W -= (self.learning_rate * dW / (np.sqrt(self.memory_W + delta)))
        self.b -= (self.learning_rate * db / (np.sqrt(self.memory_b + delta)))
        self.V -= (self.learning_rate * dV / (np.sqrt(self.memory_V + delta)))
        self.c -= (self.learning_rate * dc / (np.sqrt(self.memory_c + delta)))
        # update memory matrices
        # perform the Adagrad update of parameters

    def step(self, h0, batch_x_oh, batch_y_oh):
        h, batch_cache = self.rnn_forward(batch_x_oh, h0)
        loss, dh, dV, dc = self.output_loss_and_grads(h, batch_y_oh)
        dU, dW, db = self.rnn_backward(dh, batch_cache)
        min_val = -5
        max_val = 5
        self.update(
            np.clip(dU, min_val, max_val),
            np.clip(dW, min_val, max_val),
            np.clip(db, min_val, max_val),
            np.clip(dV, min_val, max_val),
            np.clip(dc, min_val, max_val))
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
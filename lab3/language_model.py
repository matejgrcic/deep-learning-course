import numpy as np

from RNN import RNN
from dataset import Data

def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=0.1, sample_every=100):
    
    vocab_size = len(dataset.sorted_chars)
    rnn_model = RNN(hidden_size, sequence_length, vocab_size, learning_rate) # initialize the recurrent network

    current_epoch = 0 
    batch = 0

    h0 = np.zeros((dataset.batch_size, hidden_size))

    average_loss = 0

    while current_epoch < max_epochs: 
        e, x, y = dataset.next_minibatch()
        
        if e:
            current_epoch += 1
            print(current_epoch, average_loss)
            average_loss = 0
            h0 = np.zeros((dataset.batch_size, hidden_size))
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches
        x_oh = (np.arange(vocab_size) == x[...,None]).astype(int)
        y_oh = (np.arange(vocab_size) == y[...,None]).astype(int)

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = rnn_model.step(h0, x_oh, y_oh)
        average_loss += loss / dataset.num_batches
        # if batch % sample_every == 0: 
        if e: 
            # run sampling (2.2)
            data = 'HAN:\nIs that good or bad?\n\n'
            seed = np.array(dataset.encode(data))
            output = sample(seed, 300, rnn_model)
            print(dataset.decode(output))
        batch += 1

def sample(seed, n_sample, model):
    h_initial = np.zeros((1, model.hidden_size))
    seed_onehot =  (np.arange(model.vocab_size) == seed[...,None]).astype(int)
    for i in range(seed.shape[0]):
        h_initial, _ = model.rnn_step_forward(seed_onehot[i][np.newaxis, :], h_initial)

    sample = []
    for i in range(n_sample - len(seed)):
        model_out = model.output_step(h_initial)[0]
        sample.append(np.argmax(model_out))
        h_initial, _ = model.rnn_step_forward(model_out, h_initial)

    return sample

if __name__ == '__main__':
    dataset = Data("./lab3/dataset/cornell movie-dialogs corpus/selected_conversations.txt", 10, 30)
    dataset.preprocess()
    dataset.create_minibatches()
    run_language_model(dataset, 200)
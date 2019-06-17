import numpy as np
from collections import Counter
class Data:

    def __init__(self, location, batch_size, sequence_length):
        file = open(location)
        self.data = file.read()
        file.close()

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.batch_index = -1

    def preprocess(self):
        char_counter = Counter()
        for char in list(self.data):
            char_counter[char] += 1

        self.sorted_chars = list(map(lambda x: x[0], char_counter.most_common()))

        self.char_to_id = dict(zip(self.sorted_chars, range(len(self.sorted_chars)))) 
        self.id_to_char = {k:v for v,k in self.char_to_id.items()}
        self.x = np.array(list(map(self.char_to_id.get, self.data)), dtype=np.int32)

    def encode(self, sequence):
        return [self.char_to_id[char] for char in sequence]

    def decode(self, encoded_sequence):
        return ''.join([self.id_to_char[x] for x in encoded_sequence])

    def create_minibatches(self):
        self.num_batches = len(self.x) // ((self.batch_size + self.sequence_length) * 2)
        # Is all the data going to be present in the batches? Why?
        # What happens if we select a batch size and sequence length larger than the length of the data?
        if self.num_batches < 1:
            raise Exception('Invalid number of batches')

        self.batches = []
        start_index = 0
        for i in range(self.num_batches):
            batch_x = np.zeros((self.batch_size, self.sequence_length), dtype=np.int32)
            batch_y = np.zeros((self.batch_size, self.sequence_length), dtype=np.int32)
            for j in range(self.batch_size):
                start = start_index + j
                end = start + self.sequence_length
                # print('x: ',self.decode(self.x[start: end]))
                # print('y: ',self.decode(self.x[start + 1: end + 1]))
                batch_x[j] = self.x[start: end]
                batch_y[j] = self.x[start + 1: end + 1]
            start_index += 1
            self.batches.append((batch_x, batch_y))
        return self.batches

    def next_minibatch(self):
        current = self.batch_index
        self.batch_index = (self.batch_index + 1) % self.num_batches

        batch_x, batch_y = self.batches[self.batch_index]
        new_epoch = self.batch_index < current
        return new_epoch, batch_x, batch_y

if __name__ == "__main__":
    d = Data("./lab3/dataset/cornell movie-dialogs corpus/selected_conversations.txt", 10, 30)
    d.preprocess()
    d.create_minibatches()
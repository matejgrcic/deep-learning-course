import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

class TFDeep:
    """Arguments:
       - shape: network shape
       - param_delta: training step
    """
    def __init__(self, shape, activation_type, param_delta=0.5, param_lambda=1.):
        features_count = shape[0]
        class_count = shape[-1]
        self.X = tf.placeholder(tf.float32, [None, features_count])
        self.Yoh_ = tf.placeholder(tf.float32, [None, class_count])
        self.weight_matrices = []
        self.biases = []
        self.activations = []
        
        h_previous = self.X
        for i in range(len(shape) - 1):
            W = tf.Variable(np.random.randn(shape[i], shape[i+1]), dtype=tf.float32, name=f"W_{i}")
            b = tf.Variable(np.zeros(shape[i + 1]), dtype=tf.float32, name=f"b_{i}")
            score = tf.matmul(h_previous, W) + b
            if i != len(shape) - 2:
                score = tf.nn.batch_normalization(score, 0 , 1, 0, 1, 1e-6)
            h = self._get_activation(score, activation_type, f"activation_{i}") if i != len(shape) - 2 else tf.nn.softmax(score, name=f"activation_{i}")
            self.weight_matrices.append(W)
            self.biases.append(b)
            self.activations.append(h)
            h_previous = h

        norm = 0
        for W in self.weight_matrices:
            norm += tf.norm(W)
        self.probs = h_previous
        self.loss = - tf.reduce_mean(tf.log(tf.reduce_sum(self.probs * self.Yoh_, 1))) + 0.5 * param_lambda * norm
        # optimizer = tf.train.GradientDescentOptimizer(param_delta)
        # optimizer = tf.train.AdagradOptimizer(param_delta)
        learning_rate = tf.train.exponential_decay(param_delta, 1, 1, 1-1e-4)
        optimizer = tf.train.AdagradOptimizer(learning_rate)
        self.optimization = optimizer.minimize(self.loss)
        self.train_variables = tf.trainable_variables()
        self.session = tf.Session()

    def train(self, X, Yoh_, param_niter, batch_size = 100,show_log = True):
        self.session.run(tf.global_variables_initializer())
        N = X.shape[0]
        batch_number = int(np.floor(N / batch_size))
        for i in range(param_niter):
            losses = []
            indices = np.arange(N)
            np.random.shuffle(indices)
            X = X[indices]
            Yoh_ = Yoh_[indices]
            for j in range(batch_number):
                loss, _ = self.session.run(
                    [ self.loss, self.optimization ],
                    feed_dict = {
                        self.X : X[j * batch_size : (j + 1) * batch_size],
                        self.Yoh_ : Yoh_[j * batch_size : (j + 1) * batch_size]
                    }
                )
                losses.append(loss)
            if show_log:
                print('Iteration: ', i + 1, 'Average loss: ', np.average(loss))

    def calculate_number_of_params(self, X, Yoh_):
        [variables] = self.session.run([self.train_variables], feed_dict = {self.X : X, self.Yoh_ : Yoh_})
        param_count = 0
        for variable in variables:
            if len(variable.shape) == 2:
                param_count += variable.shape[0] * variable.shape[1]
            if len(variable.shape) == 1:
                param_count += variable.shape[0]
        return param_count

    def eval(self, X):
        [probs] = self.session.run([self.probs], feed_dict = {self.X: X})
        return probs

    def _get_activation(self, score, activation_type, name):
        if activation_type == 'ReLU':
            return tf.nn.relu(score, name=name)
        elif activation_type == 'sigmoid':
            return tf.nn.sigmoid(score, name=name)
        else:
            raise TypeError('Invalid activation')

if __name__ == "__main__":
    np.random.seed(100)
    tf.set_random_seed(100)

    X,Y_ = data.sample_gmm_2d(6, 2, 10)
    Yoh_ = data.class_to_onehot(Y_)

    deep_model = TFDeep( [2, 10, 10, 2], 'sigmoid', 0.1, 1e-4)

    # nauči parametre:
    deep_model.train(X, Yoh_, 500, 5,True)

    # dohvati vjerojatnosti na skupu za učenje
    probs = deep_model.eval(X)
    Y = np.argmax(probs, axis=1)
    # # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y, Y_)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Confusion matrix:\n', confusion_matrix)

    print('Number of parameters:', deep_model.calculate_number_of_params(X, Yoh_))

    # # iscrtaj rezultate, decizijsku plohu
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    decision = lambda X: deep_model.eval(X)[:,0]
    data.graph_surface(decision, rect, offset=0)
    
    # # graph the data points
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data

class TFLogreg:
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
       - param_delta: training step
    """
    def __init__(self, D, C, param_delta=0.1, param_lambda = 1.):

        # definicija podataka i parametara:
        self.X = tf.placeholder(tf.float32, [None, D])
        self.Yoh_ = tf.placeholder(tf.float32, [None, C])
        self.W = tf.Variable(np.random.randn(D, C), dtype=tf.float32) # DxC
        self.b = tf.Variable(0.0, dtype=tf.float32)

        # formulacija modela: izračunati self.probs

        score = tf.matmul(self.X, self.W) + self.b
        self.probs = tf.nn.softmax(score)
        # formulacija gubitka: self.loss
        #   koristiti: tf.log, tf.reduce_sum, tf.reduce_mean
        # ...

        self.loss = - tf.reduce_mean(tf.log(tf.reduce_sum(self.probs * self.Yoh_, 1))) + param_lambda * 0.5 * tf.norm(self.W)

        # formulacija operacije učenja: self.train_step
        #   koristiti: tf.train.GradientDescentOptimizer,
        #              tf.train.GradientDescentOptimizer.minimize
        # ...
        optimizer = tf.train.GradientDescentOptimizer(param_delta)
        self.optimization = optimizer.minimize(self.loss)

        # instanciranje izvedbenog konteksta: self.session
        #   koristiti: tf.Session
        # ...
        self.session = tf.Session()

    """Arguments:
       - X: actual datapoints [NxD]
       - Yoh_: one-hot encoded labels [NxC]
       - param_niter: number of iterations
    """
    def train(self, X, Yoh_, param_niter, show_log = True):
    # incijalizacija parametara
    #   koristiti: tf.initialize_all_variables
    # ...
        self.session.run(tf.global_variables_initializer())
    # optimizacijska petlja
    #   koristiti: tf.Session.run
    # ...
        for i in range(param_niter):
            loss, _ = self.session.run([ self.loss, self.optimization], feed_dict = {self.X : X, self.Yoh_ : Yoh_})
            if show_log:
                print('Iteration: ', i + 1, ' Loss: ', loss)


    """Arguments:
       - X: actual datapoints [NxD]
       Returns: predicted class probabilites [NxC]
    """
    #   koristiti: tf.Session.run
    def eval(self, X):
        [probs] = self.session.run([self.probs], feed_dict = {self.X: X})
        return probs

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)
    tf.set_random_seed(100)

    # instanciraj podatke X i labele Yoh_
    X,Y_ = data.sample_gmm_2d(4, 2, 30)
    Yoh_ = data.class_to_onehot(Y_)

    # izgradi graf:
    logistic_regression_model = TFLogreg(X.shape[1], Yoh_.shape[1], 0.5, 1e-3)

    # nauči parametre:
    logistic_regression_model.train(X, Yoh_, 1000, True)

    # dohvati vjerojatnosti na skupu za učenje
    probs = logistic_regression_model.eval(X)
    Y = np.argmax(probs, axis=1)
    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y, Y_)
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Confusion matrix:\n', confusion_matrix)

    # iscrtaj rezultate, decizijsku plohu
    rect=(np.min(X, axis=0), np.max(X, axis=0))
    decision = lambda X: logistic_regression_model.eval(X)[:,0]
    data.graph_surface(decision, rect, offset=0)
    
    # graph the data points
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
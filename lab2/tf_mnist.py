import time

import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class DeepModel():
    def __init__(self,inputs, num_classes):
        # def build_model(inputs, num_classes):
        weight_decay = 1e-3
        conv1sz = 5
        mp1sz = 2
        conv2sz = 5
        mp2sz = 2
        fc3sz = 512

        _, H, W, C = inputs.shape
        self.X = tf.placeholder(tf.float32, [None, H, W, C], name='X_placeholder')
        self.Y_oh = tf.placeholder(tf.float32, [None, num_classes], name='Y_placeholder')


        net = None
        with tf.contrib.framework.arg_scope([layers.convolution2d],
            kernel_size=5, stride=1, padding='SAME', activation_fn=tf.nn.relu,
            weights_initializer=layers.variance_scaling_initializer(),
            weights_regularizer=layers.l2_regularizer(weight_decay)):

            net = layers.convolution2d(self.X, conv1sz, scope='conv1', data_format='NHWC')
            net = layers.max_pool2d(net, mp1sz, scope='mp1', data_format='NHWC')
            net = layers.convolution2d(net, conv2sz, scope='conv2', data_format='NHWC')
            net = layers.max_pool2d(net, mp2sz, scope='mp2', data_format='NHWC')

        with tf.contrib.framework.arg_scope([layers.fully_connected],
            activation_fn=tf.nn.relu,
            weights_initializer=layers.variance_scaling_initializer(),
            weights_regularizer=layers.l2_regularizer(weight_decay)):

            # sada definiramo potpuno povezane slojeve
            # ali najprije prebacimo 4D tenzor u matricu
            net = layers.flatten(net)
            net = layers.fully_connected(net, fc3sz, scope='fc3')

            self.logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')

            self.loss = tf.losses.softmax_cross_entropy(self.Y_oh, self.logits)

        optimizer = tf.train.AdamOptimizer()
        self.optimization = optimizer.minimize(self.loss)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())


    def train(self, train_x, train_y, valid_x, valid_y, config):
        lr_policy = config['lr_policy']
        batch_size = config['batch_size']
        max_epochs = config['max_epochs']
        save_dir = config['save_dir']
        num_examples = train_x.shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size

        

        for epoch in range(1, max_epochs+1):
            cnt_correct = 0

            # shuffle the data at the beggining of each epoch
            permutation_idx = np.random.permutation(num_examples)
            train_x = train_x[permutation_idx]
            train_y = train_y[permutation_idx]
            for i in range(num_batches):

                # store mini-batch to ndarray
                batch_x = train_x[i*batch_size:(i+1)*batch_size, :]
                batch_y = train_y[i*batch_size:(i+1)*batch_size, :]

                logits_val, loss_val, _ = self.session.run(
                    [self.logits, self.loss, self.optimization],
                    feed_dict={self.X: batch_x, self.Y_oh: batch_y}
                )

                # compute classification accuracy
                yp = np.argmax(logits_val, 1)
                yt = np.argmax(batch_y, 1)
                cnt_correct += (yp == yt).sum()

                if i % 5 == 0:
                    print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss_val))
                # if i % 100 == 0:
                    # draw_conv_filters(epoch, i*batch_size, net[0], save_dir)
                    #draw_conv_filters(epoch, i*batch_size, net[3])
                if i > 0 and i % 50 == 0:
                    print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
            print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
            self.evaluate("Validation", valid_x, valid_y, config)

    def evaluate(self, name, x, y, config):
        print("\nRunning evaluation: ", name)
        batch_size = config['batch_size']
        num_examples = x.shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size
        cnt_correct = 0
        loss_avg = 0
        for i in range(num_batches):
            batch_x = x[i*batch_size:(i+1)*batch_size, :]
            batch_y = y[i*batch_size:(i+1)*batch_size, :]

            logits_val, loss_val = self.session.run([self.logits, self.loss], feed_dict={self.X: batch_x, self.Y_oh: batch_y})
            yp = np.argmax(logits_val, 1)
            yt = np.argmax(batch_y, 1)
            cnt_correct += (yp == yt).sum()
            loss_avg += loss_val
            #print("step %d / %d, loss = %.2f" % (i*batch_size, num_examples, loss_val / batch_size))
        valid_acc = cnt_correct / num_examples * 100
        loss_avg /= num_batches
        print(name + " accuracy = %.2f" % valid_acc)
        print(name + " avg loss = %.2f\n" % loss_avg)
    
DATA_DIR = './datasets/MNIST/'
SAVE_DIR = "./out/"


np.random.seed(int(time.time() * 1e6) % 2**31)
dataset = input_data.read_data_sets(DATA_DIR, one_hot=True)
train_x = dataset.train.images
train_x = train_x.reshape([-1, 28, 28, 1])
train_y = dataset.train.labels
valid_x = dataset.validation.images
valid_x = valid_x.reshape([-1, 28, 28, 1])
valid_y = dataset.validation.labels
test_x = dataset.test.images
test_x = test_x.reshape([-1, 28, 28, 1])
test_y = dataset.test.labels
train_mean = train_x.mean()
train_x -= train_mean
valid_x -= train_mean
test_x -= train_mean

config = {}
config['max_epochs'] = 100
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

model = DeepModel(train_x, 10)
model.train(train_x, train_y, valid_x, valid_y, config)
model.evaluate('Test', test_x, test_y, config)

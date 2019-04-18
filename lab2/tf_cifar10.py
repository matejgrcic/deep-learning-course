import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.examples.tutorials.mnist import input_data
import math

import skimage as ski
import skimage.io

DATA_DIR = './datasets/cifar-10/'
SAVE_DIR = './out/'

def draw_image(img, mean, std):
    img *= std
    img += mean
    img = img.astype(np.uint8)
    ski.io.imshow(img)
    ski.io.show()

def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[3]
    num_channels = w.shape[2]
    k = w.shape[0]
    assert w.shape[0] == w.shape[1]
    w = w.reshape(k, k, num_channels, num_filters)
    w -= w.min()
    w /= w.max()
    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols-1) * border
    height = rows * k + (rows-1) * border
    img = np.zeros([height, width, num_channels])
    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k,c:c+k,:] = w[:,:,:,i]
    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    ski.io.imsave(os.path.join(save_dir, filename), img)


class DeepModel():
    def __init__(self,inputs, num_classes):
        weight_decay = 1e-3
        conv1sz = 16
        mp1sz = 3
        conv2sz = 32
        mp2sz = 3
        fc3sz = 256
        fc4sz = 128

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
            net = layers.fully_connected(net, fc4sz, scope='fc4')

            self.logits = layers.fully_connected(net, num_classes, activation_fn=None, scope='logits')

            # [invalid_class_logits, correct_class_logits] = tf.dynamic_partition(self.logits, tf.cast(self.Y_oh, tf.int32), 2)
            # invalid_logits = tf.reshape(invalid_class_logits, [num_classes - 1, -1])
            # self.loss = tf.reduce_mean(tf.reduce_sum(tf.maximum(invalid_logits - correct_class_logits, 0.), axis = 0))
            self.loss = tf.losses.softmax_cross_entropy(self.Y_oh, self.logits)

        self.global_step = tf.placeholder(tf.int32, [])
        self.learning_rate = tf.train.exponential_decay(0.01, self.global_step, 20, 0.96)
        optimizer = tf.train.AdagradOptimizer(self.learning_rate)
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
                batch_x = train_x[i*batch_size:(i+1)*batch_size, ...]
                batch_y = train_y[i*batch_size:(i+1)*batch_size, ...]

                logits_val, loss_val, _ = self.session.run(
                    [self.logits, self.loss, self.optimization],
                    feed_dict={self.X: batch_x, self.Y_oh: batch_y, self.global_step: epoch}
                )

                # compute classification accuracy
                yp = np.argmax(logits_val, 1)
                yt = np.argmax(batch_y, 1)
                cnt_correct += (yp == yt).sum()

                if i % 5 == 0:
                    print("epoch %d, step %d/%d, batch loss = %.2f" % (epoch, i*batch_size, num_examples, loss_val))
                if i > 0 and i % 50 == 0:
                    print("Train accuracy = %.2f" % (cnt_correct / ((i+1)*batch_size) * 100))
            print("Train accuracy = %.2f" % (cnt_correct / num_examples * 100))
            self.evaluate("Validation", valid_x, valid_y, config, epoch)
            self.evaluate("Train", train_x, train_y, config, epoch)
            
            

    def evaluate(self, name, x, y, config, epoch):
        print("\nRunning evaluation: ", name)
        batch_size = config['batch_size']
        num_examples = x.shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size
        num_classes = config['num_classes']
        confusion_matrix = np.zeros((num_classes, num_classes))
        loss_avg = 0
        for i in range(num_batches):
            batch_x = x[i*batch_size:(i+1)*batch_size, ...]
            batch_y = y[i*batch_size:(i+1)*batch_size, ...]

            logits_val, loss_val, lr_val= self.session.run(
                [self.logits, self.loss, self.learning_rate],
                feed_dict={self.X: batch_x, self.Y_oh: batch_y, self.global_step: epoch})
            yp = np.argmax(logits_val, 1)
            yt = np.argmax(batch_y, 1)
            confusion_matrix[yt, yp] += 1
            loss_avg += loss_val
        valid_acc = 0
        for i in range(num_classes):
            valid_acc += confusion_matrix[i, i]
        valid_acc /= num_examples

        accuracies = []
        for i in range(num_classes):
            print('Class:', i)
            tp = confusion_matrix[i, i]
            fp = np.sum(confusion_matrix[:, i]) - tp
            fn = np.sum(confusion_matrix[i, :]) - tp
            tn = num_examples - tp - fp - fn
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + fn + tn + fp)
            accuracies.append(accuracy)
            print(f"\tPrecision: {precision}")
            print(f"\tRecall: {recall}")
            print(f"\tAccuracy: {accuracy}")
        loss_avg /= num_batches
        acc_avg = sum(accuracies) / 10
        if name != 'Test':
          config[name + '_loss'].append(loss_avg)
          config[name + '_acc'].append(acc_avg)
        if name == 'Train':
          config['learning_rate'].append(lr_val)
        print(name + " avg accuracy = %.2f" % acc_avg)
        print(name + " avg loss = %.2f\n" % loss_avg)
        print('confusion matrix:\n', confusion_matrix)


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict

img_width = 32
img_height = 32
num_channels = 3


train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
train_y = []
for i in range(1, 6):
    subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
    train_x = np.vstack((train_x, subset['data']))
    train_y += subset['labels']
train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1)
train_y = np.array(train_y, dtype=np.int32)

subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0,2,3,1).astype(np.float32)
test_y = np.array(subset['labels'], dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)
valid_x = train_x[:valid_size, ...]
valid_y = train_y[:valid_size, ...]
train_x = train_x[valid_size:, ...]
train_y = train_y[valid_size:, ...]
data_mean = train_x.mean((0,1,2))
data_std = train_x.std((0,1,2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

y_train = np.zeros((train_y.shape[0], 10))
y_train[np.arange(train_y.shape[0]), train_y] = 1

y_valid = np.zeros((valid_y.shape[0], 10))
y_valid[np.arange(valid_y.shape[0]), valid_y] = 1

y_test = np.zeros((test_y.shape[0], 10))
y_test[np.arange(test_y.shape[0]), test_y] = 1

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}
config['num_classes'] = 10

config['Train_loss'] = []
config['Validation_loss'] = []
config['Train_acc'] = []
config['Validation_acc'] = []
config['learning_rate'] = []

model = DeepModel(train_x, 10)


conv1_var = tf.contrib.framework.get_variables('conv1/weights:0')[0]
conv1_weights = conv1_var.eval(session=model.session)
draw_conv_filters(0, 0, conv1_weights, SAVE_DIR)
# print(tf.contrib.framework.get_model_variables())
model.train(train_x, y_train, valid_x, y_valid, config)
model.evaluate('Test', test_x, y_test, config, config['max_epochs'])

conv1_var = tf.contrib.framework.get_variables('conv1/weights:0')[0]
conv1_weights = conv1_var.eval(session=model.session)
draw_conv_filters(2, 0, conv1_weights, SAVE_DIR)

losses = []
for i in range(test_x.shape[0]):
    img_loss, example_logits = model.session.run([model.loss, model.logits], feed_dict={model.X: np.array([test_x[i]]), model.Y_oh: np.array([y_test[i]])})
    losses.append((img_loss, i, test_y[i], np.argsort(-example_logits, axis=1)[:, :3]))

import matplotlib.pyplot as plt
plt.figure(figsize=(8,15))
plt.subplot(3, 1, 1)
plt.title('Average accuracy')
plt.xlabel('Iteration')
plt.plot(range(config['max_epochs']), config['Validation_acc'], label='Validation acc')
plt.plot(range(config['max_epochs']), config['Train_acc'], label='Train acc')
plt.legend()
plt.subplot(3, 1, 2)
plt.title('Cross-entropy loss')
plt.xlabel('Iteration')
plt.plot(range(config['max_epochs']), config['Validation_loss'], label='Validation loss')
plt.plot(range(config['max_epochs']), config['Train_loss'], label='Train loss')
plt.legend()
plt.subplot(3, 1, 3)
plt.title('Learning rate')
plt.xlabel('Iteration')
plt.plot(range(config['max_epochs']), config['learning_rate'], label='Learning rate')
plt.legend()
plt.show()

losses = []
for i in range(test_x.shape[0]):
    img_loss = model.session.run([model.loss], feed_dict={model.X: np.array([test_x[i]]), model.Y_oh: np.array([y_test[i]])})
    losses.append(img_loss)

losses.sort(key=lambda x: x[0], reverse=True)
class_mapper = {
    '0': 'airplane',
    '1': 'automobile',
    '2': 'bird',
    '3': 'cat',
    '4': 'deer',
    '5': 'dog',
    '6': 'frog',
    '7': 'horse',
    '8': 'ship',
    '9': 'truck'
}

for i in range(20):
    _, index, actual_class, top3_predicted = losses[i]
    print('Actual:', class_mapper[str(int(actual_class))], actual_class)
    print('Predicted')
    for x in top3_predicted[0]:
        print('\t', class_mapper[str(x)])
    draw_image(test_x[index], data_mean, data_std)

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tf_deep
import data



if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (X_train, y_train),(X_test, y_test) = mnist.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    deep_model = tf_deep.TFDeep( [784, 100, 10], 'ReLU', 0.1, 1e-4)

    # nauči parametre:
    deep_model.train(X_train, data.class_to_onehot(y_train), 100, 1000,True)

    # dohvati vjerojatnosti na skupu za učenje
    probs = deep_model.eval(X_test)
    Y = np.argmax(probs, axis=1)
    # # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y, y_test)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Confusion matrix:\n', confusion_matrix)

    print('Number of parameters:', deep_model.calculate_number_of_params(X_train, data.class_to_onehot(y_train)))
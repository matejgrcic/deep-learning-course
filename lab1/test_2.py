import tensorflow as tf
import numpy as np

X = tf.placeholder(tf.float32, [2, 2])
Y = 3 * X + 5
z = Y[0,0]
sess = tf.Session()
Y_value = sess.run(Y, feed_dict={X: [[0,1],[2,3]]})
z_value = sess.run(z, feed_dict={X: np.ones((2,2))})
print(Y_value, type(Y_value))
print(z_value, type(z_value))
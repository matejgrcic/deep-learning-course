import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. definicija računskog grafa
# podatci i parametri

X = tf.placeholder(tf.float32, [None])
Y_ = tf.placeholder(tf.float32, [None])
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# afini regresijski model

Y = a * X + b

# kvadratni gubitak

loss = ((Y-Y_)/tf.to_float(tf.shape(Y)[0])) **2
print_loss_operation = tf.print('Loss: ', loss)


N = tf.to_float(tf.shape(Y)[0])
grad_a = tf.math.reduce_sum((Y-Y_)/ N * 2 * X)
grad_b = tf.math.reduce_sum(((Y-Y_)/ N * 2))
print_grad_operation = tf.print('Grad_a: ', grad_a, 'Grad_b: ', grad_b)

# optimizacijski postupak: gradijentni spust

optimizer = tf.train.GradientDescentOptimizer(0.1)
grads_and_vars = optimizer.compute_gradients(loss, [a, b])
print_tf_calculated_grad_operation = tf.print('Grads: ', grads_and_vars)
optimize_operation = optimizer.apply_gradients(grads_and_vars)

## 2. inicijalizacija parametara

session = tf.Session()
session.run(tf.global_variables_initializer())

## 3. učenje
# neka igre počnu!

N_samples = 10
data_X = np.array([x for x in range(N_samples)])
data_Y_ = np.array([2 * x + 2 for x in range(N_samples)])

operations = [
        a,
        b,
        print_loss_operation,
        optimize_operation,
        print_grad_operation,
        print_tf_calculated_grad_operation,
    ]
for i in range(100):
    print('Iteration: ', i)
    val_a, val_b, _, _, _, _ = session.run(operations, feed_dict={X: data_X, Y_: data_Y_})

plt.plot(data_X, data_X * val_a + val_b, 'r--')
plt.scatter(data_X, data_Y_)
plt.show()
import numpy as np
import tensorflow as tf

# oblikovanje računskog grafa
a = tf.constant(5)
b = tf.constant(8)
x = tf.placeholder(dtype='int32')
c = a + b * x
d = b * x

# fazu zadavanja upita započinjemo 
# stvaranjem izvedbenog konteksta:
session = tf.Session()

# zadajemo upit: izračunati c uz x=5
c_val = session.run(c, feed_dict={x: 5})

# ispis rezultata
print(c_val)
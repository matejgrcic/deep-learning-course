import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import tile_raster_images
import math
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'

mnist = input_data.read_data_sets("./lab4/dataset/MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images,\
    mnist.test.labels

def weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias(shape):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial)

def sample_prob(probs):
    """Uzorkovanje vektora x prema vektoru vjerojatnosti p(x=1) = probs"""
    return tf.to_float(tf.random_uniform(tf.shape(probs)) <= probs)

def draw_weights(W, shape, N, stat_shape, interpolation="bilinear"):
    """Vizualizacija težina
    W -- vektori težina
    shape -- tuple dimenzije za 2D prikaz težina - obično dimenzije ulazne slike, npr. (28,28)
    N -- broj vektora težina
    shape_state -- dimezije za 2D prikaz stanja (npr. za 100 stanja (10,10)
    """
    image = (tile_raster_images(
        X=W.T,
        img_shape=shape,
        tile_shape=(int(math.ceil(N/stat_shape[0])), stat_shape[0]),
        tile_spacing=(1, 1)))
    plt.figure(figsize=(10, 14))
    plt.imshow(image, interpolation=interpolation)
    plt.axis('off')

def draw_reconstructions(ins, outs, states, shape_in, shape_state, N):
    """Vizualizacija ulaza i pripadajućih rekonstrukcija i stanja skrivenog sloja
    ins -- ualzni vektori
    outs -- rekonstruirani vektori
    states -- vektori stanja skrivenog sloja
    shape_in -- dimezije ulaznih slika npr. (28,28)
    shape_state -- dimezije za 2D prikaz stanja (npr. za 100 stanja (10,10)
    N -- broj uzoraka
    """
    plt.figure(figsize=(8, int(2 * N)))
    for i in range(N):
        plt.subplot(N, 4, 4*i + 1)
        plt.imshow(ins[i].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Test input")
        plt.axis('off')
        plt.subplot(N, 4, 4*i + 2)
        plt.imshow(outs[i][0:784].reshape(shape_in), vmin=0, vmax=1, interpolation="nearest")
        plt.title("Reconstruction")
        plt.axis('off')
        plt.subplot(N, 4, 4*i + 3)
        plt.imshow(states[i].reshape(shape_state), vmin=0, vmax=1, interpolation="nearest")
        plt.title("States")
        plt.axis('off')
    plt.tight_layout()


Nh = 100 # Broj elemenata prvog skrivenog sloja
h1_shape = (10,10)
Nv = 784 # Broj elemenata vidljivog sloja
v_shape = (28,28)
Nu = 5000 # Broj uzoraka za vizualizaciju rekonstrukcije

gibbs_sampling_steps = 1
alpha = 0.1

g1 = tf.Graph()
with g1.as_default():
        
    X1 = tf.placeholder("float", [None, 784])
    w1 = weights([Nv, Nh])
    vb1 = bias([Nv])
    hb1 = bias([Nh])

    h0_prob = tf.sigmoid(tf.matmul(X1, w1) + hb1)
    h0 = sample_prob(h0_prob)
    h1 = h0

    for step in range(gibbs_sampling_steps):
        v1_prob = tf.sigmoid(tf.transpose(tf.matmul(w1, tf.transpose(h1))) + vb1)
        v1 = sample_prob(v1_prob)
        h1_prob = tf.sigmoid(tf.matmul(v1, w1) + hb1)
        h1 = sample_prob(h1_prob)
        
    
    w1_positive_grad = tf.matmul(tf.transpose(X1), h0_prob)
    w1_negative_grad = tf.matmul(tf.transpose(v1_prob), h1_prob)

    dw1 = (w1_positive_grad - w1_negative_grad) / tf.to_float(tf.shape(X1)[0])

    update_w1 = tf.assign_add(w1, alpha * dw1)
    update_vb1 = tf.assign_add(vb1, alpha * tf.reduce_mean(X1 - v1, 0))
    update_hb1 = tf.assign_add(hb1, alpha * tf.reduce_mean(h0 - h1, 0)) 

    out1 = (update_w1, update_vb1, update_hb1)
    
    v1_prob = tf.sigmoid(tf.transpose(tf.matmul(w1, tf.transpose(h1))) + vb1)
    v1 = sample_prob(v1_prob)
    
    err1 = X1 - v1_prob
    err_sum1 = tf.reduce_mean(err1 * err1)
    
    initialize1 = tf.global_variables_initializer()

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples
total_batch = int(n_samples / batch_size) * epochs

sess1 = tf.Session(graph=g1)
sess1.run(initialize1)

for i in range(total_batch):
    batch, label = mnist.train.next_batch(batch_size)
    err, _ = sess1.run([err_sum1, out1], feed_dict={X1: batch})
        
    if i%(int(total_batch/10)) == 0:
        print(i, err)

w1s = w1.eval(session=sess1)
vb1s = vb1.eval(session=sess1)
hb1s = hb1.eval(session=sess1)
vr, h1s = sess1.run([v1_prob, h1], feed_dict={X1: teX[0:Nu,:]})

Nh2 = Nh + 10 # Broj elemenata drugog skrivenog sloja
h2_shape = h1_shape 

gibbs_sampling_steps = 2
alpha = 0.1

g2 = tf.Graph()
with g2.as_default():
    
    X2 = tf.placeholder("float", [None, Nv])
    w1a = tf.Variable(w1s)
    vb1a = tf.Variable(vb1s)
    hb1a = tf.Variable(hb1s)
    w2 = weights([Nh, Nh2])
    hb2 = bias([Nh2])
    
    h1up_prob = tf.sigmoid(tf.matmul(X2, w1a) + hb1a)
    h1up = sample_prob(h1up_prob)
    h2up_prob = tf.sigmoid(tf.matmul(h1up, w2) + hb2)
    h2up = sample_prob(h2up_prob)
    h2down = h2up
    
    for step in range(gibbs_sampling_steps):
        h1down_prob = tf.sigmoid(tf.matmul(h2down, tf.transpose(w2)) + hb1a)
        h1down = sample_prob(h1down_prob)
        h2down_prob = tf.sigmoid(tf.matmul(h1down, w2) + hb2)
        h2down = sample_prob(h2down_prob)
    
    w2_positive_grad = tf.matmul(tf.transpose(h1up), h2up_prob)
    w2_negative_grad = tf.matmul(tf.transpose(h1down_prob), h2down_prob)


    dw2 = (w2_positive_grad - w2_negative_grad) / tf.to_float(tf.shape(h1up)[0])

    update_w2 = tf.assign_add(w2, alpha * dw2)
    update_hb1a = tf.assign_add(hb1a, alpha * tf.reduce_mean(h1up - h1down, 0))
    update_hb2 = tf.assign_add(hb2, alpha * tf.reduce_mean(h2up - h2down, 0))

    out2 = (update_w2, update_hb1a, update_hb2)

    # rekonsturkcija ulaza na temelju krovnog skrivenog stanja h3
    # ...
    # ...
    v_out_prob1 = tf.sigmoid(tf.matmul(h2down, tf.transpose(w2)) + hb1a)
    v_out1 = sample_prob(v_out_prob1)
    v_out_prob = tf.sigmoid(tf.matmul(v_out1, tf.transpose(w1a)) + vb1a)
    v_out = sample_prob(v_out_prob)
    
    err2 = X2 - v_out_prob
    err_sum2 = tf.reduce_mean(err2 * err2)
    
    initialize2 = tf.global_variables_initializer()

batch_size = 100
epochs = 100
n_samples = mnist.train.num_examples

total_batch = int(n_samples / batch_size) * epochs

sess2 = tf.Session(graph=g2)
sess2.run(initialize2)
for i in range(total_batch):
    # iteracije treniranja 
    batch, label = mnist.train.next_batch(batch_size)
    err, _ = sess2.run([err_sum2, out2], feed_dict={X2: batch})
    if i%(int(total_batch/10)) == 0:
        print(i, err)
        
    w2s, hb1as, hb2s = sess2.run([w2, hb1a, hb2], feed_dict={X2: batch})
    vr2, h2downs = sess2.run([v_out_prob, h2down], feed_dict={X2: teX[0:Nu,:]})

# vizualizacija težina
draw_weights(w2s, h1_shape, Nh2, h2_shape, interpolation="nearest")

# vizualizacija rekonstrukcije i stanja
draw_reconstructions(teX, vr2, h2downs, v_shape, h2_shape, 200)

# Generiranje uzoraka iz slučajnih vektora krovnog skrivenog sloja
#...
#...
# Emulacija dodatnih Gibbsovih uzorkovanja pomoću feed_dict
#...
#...
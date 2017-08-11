import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.examples.tutorials.mnist import input_data
from collections import defaultdict
from collections import Counter

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist)

class Kmeans():
    def __init__(self,cluster,input):
        self.cluster = cluster
        self.input = input
    def run(self):
        cluster = KMeans(n_clusters=self.cluster, random_state=0)
        kmeans = cluster.fit(self.input)
        return kmeans.labels_

#define conv
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#define max pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#define weight variable for padding
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

#define bias for relu
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def checksize(layer,a,b,c):
    print(layer+"----------")
    print(a)
    print(b)
    print(c)

def get_cluster(arr):
    val = arr[250]
    set = []
    for i in range(250):
        if arr[i]==val:
            set.append(i)
    return set

xs = tf.placeholder(tf.float32, [40, 40])   # 28x28
x_image = tf.reshape(xs, [-1, 40, 40, 1])

#first conv layer  input 40*40*1 - padding 40*40*32 - pooling 20*20*32
W1 = weight_variable([5,5, 1,32])
b1 = bias_variable([32])
conv1 = tf.nn.relu(conv2d(x_image, W1) + b1)
pool1 = max_pool_2x2(conv1)
print(pool1.shape)


#second conv layer input 20*20*32 -padding 20*20*64 - pooling 10*10*64
W2 = weight_variable([5,5, 32,64])
b2 = bias_variable([64])
conv2 = tf.nn.relu(conv2d(pool1, W2) + b2)
pool2 = max_pool_2x2(conv2)
print(pool2.shape)

#third conv layer input 10*10*64 -padding 10*10*96 - pooling 5*5*96
W3 = weight_variable([5,5, 64,96])
b3 = bias_variable([96])
conv3 = tf.nn.relu(conv2d(pool2, W3) + b3)
pool3 = max_pool_2x2(conv3)
print(pool3.shape)


#fully connected layer input 5*5*96 - flat 3096*1
W4 = weight_variable([5*5*96, 1000])
b4 = bias_variable([1000])
flat = tf.reshape(pool3, [1, 5*5*96])
fl = tf.nn.relu(tf.matmul(flat, W4) + b4)
#fl = tf.nn.dropout(fl, 0.5)


#main control
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
'''
matrix1 = np.random.rand(40,40)
print(sess.run(fl,feed_dict={xs:matrix1}))
'''
data = np.load("list.npy")
right = np.load("compare.npy")
kmeans_input = []
print(right.shape)
for i in range(250):
    input = data[i]
    vector = sess.run(fl, feed_dict={xs: input})
    kmeans_input.append(vector[0])
right_vector = sess.run(fl,feed_dict={xs:right})

right_vector = right_vector[0]

bio_data = {}

kmeans_input.append(right_vector)

#print(kmeans_input)
res = Kmeans(10,kmeans_input).run()
print(get_cluster(res))
res = Kmeans(20,kmeans_input).run()
print(get_cluster(res))
res = Kmeans(30,kmeans_input).run()
print(get_cluster(res))

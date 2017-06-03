import numpy as np
import tensorflow as tf

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

#first conv layer  input 28*28*1 - padding 28*28*32 - pooling 14*14*32
def conv_layer1(input):
    W = weight_variable([5,5, 1,32])
    b = bias_variable([32])
    conv = tf.nn.relu(conv2d(input, W) + b)
    pool = max_pool_2x2(conv)
    checksize("conv_layer1",input.shape,conv.shape,pool.shape)
    return pool

#second conv layer input 14*14*32 -padding 14*14*64 - pooling 7*7*64
def conv_layer2(input):
    W = weight_variable([5,5, 32,64])
    b = bias_variable([64])
    conv = tf.nn.relu(conv2d(input, W) + b)
    pool = max_pool_2x2(conv)
    checksize("conv_layer2",input.shape, conv.shape, pool.shape)
    return pool


#fully connected layer input 7*7*64 - flat 3096*1 dropoout 500*1
def fl_layer(input,size):
    W = weight_variable([7*7*64, size])
    b = bias_variable([size])
    flat = tf.reshape(input, [1, 7*7*64])
    fl = tf.nn.dropout(tf.nn.relu(tf.matmul(flat, W) + b),0.5)
    checksize("fully_connected_layer1",input.shape,flat.shape,fl.shape)
    return fl

#main control
def run(input,size):
    p1 = conv_layer1(input)
    p2 = conv_layer2(p1)
    fl = fl_layer(p2,size)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    res = sess.run(fl)
    return res


#test
input = np.float32(np.random.randint(0, 100, size=(28,28)))
print(input)
input = tf.reshape(input, [-1, 28, 28, 1])

res = run(input,1000)
print(res[0])
print(len(res[0]))

#next use this to do the k-means

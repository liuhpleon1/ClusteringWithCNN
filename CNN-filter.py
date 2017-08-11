import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from tensorflow.examples.tutorials.mnist import input_data
from collections import defaultdict
from collections import Counter

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#print(mnist)

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

xs = tf.placeholder(tf.float32, [None, 784])/255.   # 28x28
x_image = tf.reshape(xs, [-1, 28, 28, 1])

#first conv layer  input 28*28*1 - padding 28*28*32 - pooling 14*14*32
W1 = weight_variable([5,5, 1,32])
b1 = bias_variable([32])
conv1 = tf.nn.relu(conv2d(x_image, W1) + b1)
pool1 = max_pool_2x2(conv1)
#checksize("conv_layer1",input.shape,conv.shape,pool.shape)


#second conv layer input 14*14*32 -padding 14*14*64 - pooling 7*7*64
W2 = weight_variable([5,5, 32,64])
b2 = bias_variable([64])
conv2 = tf.nn.relu(conv2d(pool1, W2) + b2)
pool2 = max_pool_2x2(conv2)
#checksize("conv_layer2",input.shape, conv.shape, pool.shape)



#fully connected layer input 7*7*64 - flat 3096*1 dropoout 500*1
W3 = weight_variable([7*7*64, 1000])
b3 = bias_variable([1000])
flat = tf.reshape(pool2, [1, 7*7*64])
fl = tf.nn.relu(tf.matmul(flat, W3) + b3)
#fl = tf.nn.dropout(fl, 0.5)
#checksize("fully_connected_layer1",input.shape,flat.shape,fl.shape)

#main control
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)



#test minist
size = 200
input1 = []
input2 = []
real = []
for i in range(size):
    batch_xs,batch_ys = mnist.train.next_batch(1)
    batch_ys = batch_ys.reshape(10)
    image = batch_xs.reshape(28,28)
    #print(image)
    for i in range(28):
        for j in range(28):
            if image[i][j]>0:
                image[i][j] = 1
    image = np.int32(image)
    realnum = 0
    for i in range(10):
        if batch_ys[i] == 1:
            realnum = i
            break
    #print(image)
    '''
    print(write)
    print(realnum)
    '''
    real.append(realnum)

    kmeans_input_e2 = batch_xs.reshape(1,784)
    cnn_output= sess.run(fl,feed_dict={xs:batch_xs})
    kmeans_input_e1 = cnn_output
    '''
    print(kmeans_input_e1[0])
    print(kmeans_input_e2[0])
    '''

    input1.append(kmeans_input_e1[0])
    input2.append(kmeans_input_e2[0])
    #print(cnn_output)
'''
print(input1)
print(input2)
'''
#next use this to do the k-means

res_without_cnn = Kmeans(10,input2).run()
res_with_cnn = Kmeans(10,input1).run()
'''
print(res_without_cnn)
print(res_with_cnn)
print(real)
'''
c1 = defaultdict(list)
c2 = defaultdict(list)

for i in range(size):
    c1[res_without_cnn[i]].append(real[i])
    c2[res_with_cnn[i]].append(real[i])

a1 = 0
a2 = 0

for i in range(10):
    list1 = c1[i]
    list2 = c2[i]
    counts1 = Counter(list1)
    most_common1 = counts1.most_common(1)
    counts2 = Counter(list2)
    most_common2 = counts2.most_common(1)
    a1 = a1+0.1*(most_common1[0][1]/len(list1))
    a2 = a2+0.1*(most_common2[0][1]/len(list2))

print(c1)
print("Kmeans")
print(a1)
print(c2)
print("CNN+Kmeans")
print(a2)


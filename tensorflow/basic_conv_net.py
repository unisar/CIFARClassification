import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split

#load data
input = np.load('X_train.npy')   
labels = np.genfromtxt('../data/y_train.txt')

input = np.transpose(input,(0,2,3,1))

X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=0.1, random_state=42, stratify=labels)

def one_hot(y):
    retVal = np.zeros((len(y), 10))
    retVal[np.arange(len(y)), y.astype(int)] = 1
    return retVal

y_train = one_hot(y_train)
y_test = one_hot(y_test)

#define convolutional neural network model
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None,32,32,3])
y_ = tf.placeholder(tf.float32, shape=[None,10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
  
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([3, 3, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

W_conv2 = weight_variable([3, 3, 32, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([3, 3, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

W_conv4 = weight_variable([3, 3, 64, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = weight_variable([3, 3, 64, 128])
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)

W_conv6 = weight_variable([3, 3, 128, 128])
b_conv6 = bias_variable([128])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

W_fc1 = weight_variable([8 * 8 * 128, 2000])
b_fc1 = bias_variable([2000])

h_pool_flat = tf.reshape(h_conv6, [-1, 8*8*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([2000, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#run session

batch_size = 200

sess.run(tf.initialize_all_variables())
for i in range(30000):
    indices = np.random.permutation(X_train.shape[0])[:batch_size]
    X_batch = X_train[indices,:,:,:]
    y_batch = y_train[indices,:]
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:X_batch, y_:y_batch, keep_prob: 1.0})
        print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x:X_batch, y_:y_batch, keep_prob: 0.5})

print "test accuracy %g" % accuracy.eval(feed_dict={x:X_test, y_:y_test, keep_prob: 1.0})
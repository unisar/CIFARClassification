import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import random

def one_hot(y):
    retVal = np.zeros((len(y), 10))
    retVal[np.arange(len(y)), y.astype(int)] = 1
    return retVal

#load data
input = np.load('X_train_zca.npy')   
input = input.transpose(0,2,3,1)
labels = np.genfromtxt('../data/y_train.txt')

X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=0.1, random_state=42, stratify=labels)

y_train = one_hot(y_train)
y_test = one_hot(y_test)

X_full = np.load('X_test_zca.npy')
X_full = X_full.transpose(0,2,3,1)

noOfIterations = 180000
image_size = 32
num_channels = 3
num_labels=10
batch_size = 100
patch_size = 3
open('accuracy.txt', 'w').close()

sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return initial

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return initial

def conv2d(x, W, stride=[1,1,1,1],pad='SAME'):
    return tf.nn.conv2d(x, W, strides=stride, padding=pad)
	
def fmaxpool(x):
    h_pool, rows, cols = tf.nn.fractional_max_pool(x,[1.0, 1.44, 1.44, 1.0],True,True)
    return h_pool

tfx = tf.placeholder(tf.float32, shape=[None,image_size,image_size,num_channels])
tfy = tf.placeholder(tf.float32, shape=[None,num_labels])

w1 = tf.Variable(weight_variable([3,3,3,96]))
b1 = tf.Variable(bias_variable([96]))
l1 = tf.nn.elu(conv2d(tfx,w1) + b1)

w2 = tf.Variable(weight_variable([3,3,96,96]))
b2 = tf.Variable(bias_variable([96]))
l2 = tf.nn.elu(conv2d(l1, w2) + b2)
mp2 = fmaxpool(l2)

w3 = tf.Variable(weight_variable([3,3,96,140]))
b3 =  tf.Variable(bias_variable([140]))
l3 = tf.nn.elu(conv2d(mp2, w3) + b3)

w4 = tf.Variable(weight_variable([3,3,140,140]))
b4 =  tf.Variable(bias_variable([140]))
l4 = tf.nn.elu(conv2d(l3, w4) + b4)
mp4 = fmaxpool(l4)

w5 = tf.Variable(weight_variable([3,3,140,200]))
b5 =  tf.Variable(bias_variable([200]))
l5 = tf.nn.elu(conv2d(mp4, w5) + b5)

w6 = tf.Variable(weight_variable([3,3,200,200]))
b6 =  tf.Variable(bias_variable([200]))
l6 = tf.nn.elu(conv2d(l5, w6) + b6)
mp6 = fmaxpool(l6)

w7 = tf.Variable(weight_variable([3,3,200,256]))
b7 =  tf.Variable(bias_variable([256]))
l7 = tf.nn.elu(conv2d(mp6, w7) + b7)

w8 = tf.Variable(weight_variable([1,1,256,256]))
b8 =  tf.Variable(bias_variable([256]))
l8 = tf.nn.elu(conv2d(l7, w8) + b8)

w9 = tf.Variable(weight_variable([1,1,256,10]))
b9 =  tf.Variable(bias_variable([10]))
l9 = tf.nn.elu(conv2d(l8, w9) + b9)

avg = tf.nn.avg_pool(l9,[1,10,10,1],[1,1,1,1],'VALID')
reshape = tf.reshape(avg, [-1, 10])
w10 = tf.Variable(weight_variable([10, num_labels]))
b10 = tf.Variable(bias_variable([num_labels]))
lastLayer = tf.matmul(reshape, w10) + b10

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer,tfy))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
prediction = tf.nn.softmax(lastLayer)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(tfy,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

sess.run(init_op)
for i in range(noOfIterations):
    indices = np.random.permutation(X_train.shape[0])[:batch_size]
    X_batch = X_train[indices,:,:,:]
    crop1 = np.random.randint(-5,6)
    crop2 = np.random.randint(-5,6)
    if crop1 > 0:
        X_batch = np.concatenate((X_batch[:,crop1:,:,:],np.zeros((batch_size,crop1,image_size,num_channels))),axis=1)
    elif crop1 < 0:
        X_batch = np.concatenate((np.zeros((batch_size,-crop1,image_size,num_channels)),X_batch[:,:crop1,:,:]),axis=1)
    if crop2 > 0:
        X_batch = np.concatenate((X_batch[:,:,crop2:,:],np.zeros((batch_size,image_size,crop2,num_channels))),axis=2)
    elif crop2 < 0:
        X_batch = np.concatenate((np.zeros((batch_size,image_size,-crop2,num_channels)),X_batch[:,:,:crop2,:]),axis=2)   
    y_batch = y_train[indices,:]
    if random.random() < .5:
        X_batch = np.fliplr(X_batch)
        y_batch = np.flipud(y_batch)
    feed_dict = {tfx:X_batch,tfy:y_batch}
    _ = sess.run(optimizer, feed_dict=feed_dict)

    if (i % 100 == 0):
        test_accuracies = []
        for j in range(0,X_test.shape[0],batch_size):
            feed_dict={tfx:X_test[j:j+batch_size,:,:,:],tfy:y_test[j:j+batch_size,:]}
            test_accuracies.append(sess.run(accuracy, feed_dict=feed_dict)*100)
        print 'iteration %i test accuracy: %.4f%%' % (i, np.mean(test_accuracies))
        with open("accuracy.txt", "a") as f:
            f.write('iteration %i test accuracy: %.4f%%\n' % (i, np.mean(test_accuracies)))
        
    if (i % 10000 == 0):
        preds = []
        for j in range(0,X_full.shape[0],batch_size):
            feed_dict={tfx:X_full[j:j+batch_size,:,:,:]}
            p = sess.run(prediction, feed_dict=feed_dict)
            preds.append(np.argmax(p, 1))
        pred = np.concatenate(preds)
        np.savetxt('prediction.txt',pred,fmt='%.0f')
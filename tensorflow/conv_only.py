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
regularization = 0.001
open('accuracy.txt', 'w').close()

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return initial

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return initial

def conv2d(x, W, stride=[1,1,1,1],pad='SAME'):
    return tf.nn.conv2d(x, W, strides=stride, padding=pad)

tfx = tf.placeholder(tf.float32, shape=[None,image_size,image_size,num_channels])
tfy = tf.placeholder(tf.float32, shape=[None,num_labels])
kp0 = tf.placeholder(tf.float32)
d0 = tf.nn.dropout(tfx, kp0)

w1 = tf.Variable(weight_variable([3,3,3,96]))    
b1 = tf.Variable(bias_variable([96]))
l1 = tf.nn.elu(conv2d(d0,w1) + b1)

w2 = tf.Variable(weight_variable([3,3,96,96]))
b2 = tf.Variable(bias_variable([96]))    
l2 = tf.nn.elu(conv2d(l1, w2) + b2)

w3 = tf.Variable(weight_variable([3,3,96,96]))
b3 = tf.Variable(bias_variable([96]))
l3 = tf.nn.elu(conv2d(l2, w3, [1,2,2,1]) + b3)
kp3 = tf.placeholder(tf.float32)
d3 = tf.nn.dropout(l3, kp3)

w4 = tf.Variable(weight_variable([3,3,96,192]))
b4 =  tf.Variable(bias_variable([192]))
l4 = tf.nn.elu(conv2d(d3, w4) + b4)

w5 = tf.Variable(weight_variable([3,3,192,192]))
b5 =  tf.Variable(bias_variable([192]))
l5 = tf.nn.elu(conv2d(l4, w5) + b5)

w6 = tf.Variable(weight_variable([3,3,192,192]))
b6 =  tf.Variable(bias_variable([192]))
l6 = tf.nn.elu(conv2d(l5, w6, [1,2,2,1]) + b6)
kp6 = tf.placeholder(tf.float32)
d6 = tf.nn.dropout(l6, kp6)

w7 = tf.Variable(weight_variable([3,3,192,192]))
b7 =  tf.Variable(bias_variable([192]))
l7 = tf.nn.elu(conv2d(d6, w7, [1,1,1,1]) + b7)

w8 = tf.Variable(weight_variable([1,1,192,192]))
b8 =  tf.Variable(bias_variable([192]))
l8 = tf.nn.elu(conv2d(l7, w8, [1,1,1,1]) + b8)

w9 = tf.Variable(weight_variable([1,1,192,10]))
b9 =  tf.Variable(bias_variable([10]))
l9 = tf.nn.elu(conv2d(l8, w9, [1,1,1,1]) + b9)

avg = tf.nn.avg_pool(l9,[1,8,8,1],[1,1,1,1],'VALID')
reshape = tf.reshape(avg, [-1, 10])
w10 = tf.Variable(weight_variable([10, num_labels]))
b10 = tf.Variable(bias_variable([num_labels]))
lastLayer = tf.matmul(reshape, w10) + b10

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer,tfy) + 
    regularization * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) +
    tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5) + tf.nn.l2_loss(w6) + tf.nn.l2_loss(w7) + 
    tf.nn.l2_loss(w8) + tf.nn.l2_loss(w9) + tf.nn.l2_loss(w10)))
lr = tf.placeholder(tf.float32)
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
prediction = tf.nn.softmax(lastLayer)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(tfy,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

sess.run(init_op)
learning_rate = 0.1
for i in range(noOfIterations):
    if (i+1) % 200 == 0:
        learning_rate *= .99
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
        X_batch = X_batch[:,:,::-1,:]
    feed_dict = {tfx:X_batch,tfy:y_batch,kp0:0.8,kp3:0.5,kp6:0.5,lr:learning_rate}
    _ = sess.run(optimizer, feed_dict=feed_dict)

    if (i % 100 == 0):
        test_accuracies = []
        for j in range(0,X_test.shape[0],batch_size):
            feed_dict={tfx:X_test[j:j+batch_size,:,:,:],tfy:y_test[j:j+batch_size,:],kp0:1.0,kp3:1.0,kp6:1.0}
            test_accuracies.append(sess.run(accuracy, feed_dict=feed_dict)*100)
        print 'iteration %i test accuracy: %.4f%%' % (i, np.mean(test_accuracies))
        with open("accuracy.txt", "a") as f:
            f.write('iteration %i test accuracy: %.4f%%\n' % (i, np.mean(test_accuracies)))
        
    if (i % 10000 == 0):
        preds = []
        for j in range(0,X_full.shape[0],batch_size):
            feed_dict={tfx:X_full[j:j+batch_size,:,:,:],kp0:1.0,kp3:1.0,kp6:1.0}
            p = sess.run(prediction, feed_dict=feed_dict)
            preds.append(np.argmax(p, 1))
        pred = np.concatenate(preds)
        np.savetxt('prediction.txt',pred,fmt='%.0f')
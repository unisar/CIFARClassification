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

noOfIterations = 180000
image_size = 24
num_channels = 3
num_labels=10
batch_size = 100
patch_size = 3

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return initial

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return initial

def conv2d(x, W, stride=[1,1,1,1],pad='SAME'):
    return tf.nn.conv2d(x, W, strides=stride, padding=pad)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

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

avg = tf.nn.avg_pool(l9,[1,6,6,1],[1,1,1,1],'VALID')
reshape = tf.reshape(avg, [-1, 10])
w10 = tf.Variable(weight_variable([10, num_labels]))
b10 = tf.Variable(bias_variable([num_labels]))
lastLayer = tf.matmul(reshape, w10) + b10

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer,tfy))
lr = tf.placeholder(tf.float32)
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
prediction=tf.nn.softmax(lastLayer)

model_saver = tf.train.Saver({  'w1': w1, 'b1': b1,
                                'w2': w2, 'b2': b2,
                                'w3': w3, 'b3': b3,
                                'w4': w4, 'b4': b4,
                                'w5': w5, 'b5': b5,
                                'w6': w6, 'b6': b6,
                                'w7': w7, 'b7': b7, 
                                'w8': w8, 'b8': b8,
                                'w9': w9, 'b9': b9,
                                'w10': w10, 'b10': b10
                            })

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

sess.run(init_op)
learning_rate = 0.1
for i in range(noOfIterations):
    if i==100000:
        learning_rate = 0.01
    if i==125000:
        learning_rate = 0.001
    if i==150000:
        learning_rate = 0.0001
    indices = np.random.permutation(X_train.shape[0])[:batch_size]
    crop1 = np.random.randint(0,8)
    crop2 = np.random.randint(0,8)
    X_batch = X_train[indices,crop1:crop1+24,crop2:crop2+24,:]
    y_batch = y_train[indices,:]
    if random.random() < .5:
        X_batch = np.fliplr(X_batch)
        y_batch = np.flipud(y_batch)
    feed_dict = {tfx:X_batch,tfy:y_batch,kp0:0.2,kp3:0.5,kp6:0.5,lr:learning_rate}
    _, l, predictions = sess.run([optimizer, loss, prediction], feed_dict=feed_dict)

    if (i % 100 == 0):
        test_accuracies = []
        for j in range(0,X_test.shape[0],batch_size):
            feed_dict={tfx:X_test[j:j+batch_size,4:28,4:28,:],tfy:y_test[j:j+batch_size,:],kp0:1.0,kp3:1.0,kp6:1.0}
            l, predictions = sess.run([loss, prediction], feed_dict=feed_dict)
            test_accuracies.append(accuracy(predictions,y_test[j:j+batch_size,:]))
        print 'iteration %i test accuracy: %.1f%%' % (i, np.mean(test_accuracies))
        
    if (i % 5000 == 0):
        print "Saving the model"
        model_saver.save(sess, 'model.ckpt')
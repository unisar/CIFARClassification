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

noOfIterations = 100000
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

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x,stride):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=stride, padding='SAME')

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

tfx = tf.placeholder(tf.float32, shape=[None,image_size,image_size,num_channels])
tfy = tf.placeholder(tf.float32, shape=[None,num_labels])

w1 = tf.Variable(weight_variable([3,3,3,40]))    
b1 = tf.Variable(bias_variable([40]))
l1 = tf.nn.elu(conv2d(tfx,w1) + b1)
mean1,var1 = tf.nn.moments(l1,[0])
beta1 = tf.Variable(tf.zeros([1,24,24,40]))
scale1 = tf.Variable(tf.ones([1,24,24,40]))
bn1 = tf.nn.batch_normalization(l1,mean1,var1,beta1,scale1,0.000001)

w2 = tf.Variable(weight_variable([3,3,40,40]))
b2 = tf.Variable(bias_variable([40]))    
l2 = tf.nn.elu(conv2d(bn1, w2) + b2)
mean2,var2 = tf.nn.moments(l2,[0])
beta2 = tf.Variable(tf.zeros([1,24,24,40]))
scale2 = tf.Variable(tf.ones([1,24,24,40]))
bn2 = tf.nn.batch_normalization(l2,mean2,var2,beta2,scale2,0.000001)
maxpool1 = max_pool_2x2(bn2, [1, 2, 2, 1])

w3 = tf.Variable(weight_variable([3,3,40,80]))
b3 = tf.Variable(bias_variable([80]))
l3 = tf.nn.elu(conv2d(maxpool1, w3) + b3)
mean3,var3 = tf.nn.moments(l3,[0])
beta3 = tf.Variable(tf.zeros([1,12,12,80]))
scale3 = tf.Variable(tf.ones([1,12,12,80]))
bn3 = tf.nn.batch_normalization(l3,mean3,var3,beta3,scale3,0.000001)

w4 = tf.Variable(weight_variable([3,3,80,80]))
b4 =  tf.Variable(bias_variable([80]))
l4 = tf.nn.elu(conv2d(bn3, w4) + b4)
mean4,var4 = tf.nn.moments(l4,[0])
beta4 = tf.Variable(tf.zeros([1,12,12,80]))
scale4 = tf.Variable(tf.ones([1,12,12,80]))
bn4 = tf.nn.batch_normalization(l4,mean4,var4,beta4,scale4,0.000001)
maxpool2 = max_pool_2x2(bn4, [1, 2, 2, 1])

w5 = tf.Variable(weight_variable([3,3,80,160]))
b5 =  tf.Variable(bias_variable([160]))
l5 = tf.nn.elu(conv2d(maxpool2, w5) + b5)
mean5,var5 = tf.nn.moments(l5,[0])
beta5 = tf.Variable(tf.zeros([1,6,6,160]))
scale5 = tf.Variable(tf.ones([1,6,6,160]))
bn5 = tf.nn.batch_normalization(l5,mean5,var5,beta5,scale5,0.000001)

w6 = tf.Variable(weight_variable([3,3,160,160]))
b6 =  tf.Variable(bias_variable([160]))
l6 = tf.nn.elu(conv2d(bn5, w6) + b6)
mean6,var6 = tf.nn.moments(l6,[0])
beta6 = tf.Variable(tf.zeros([1,6,6,160]))
scale6 = tf.Variable(tf.ones([1,6,6,160]))
bn6 = tf.nn.batch_normalization(l6,mean6,var6,beta6,scale6,0.000001)

shape = bn6.get_shape().as_list()
w7 = tf.Variable(weight_variable([6 * 6 * 160, 2000]))
b7 = tf.Variable(bias_variable([2000]))
reshape = tf.reshape(bn6, [-1, shape[1] * shape[2] * shape[3]])
l7 = tf.nn.elu(tf.matmul(reshape, w7) + b7)

keep_prob = tf.placeholder(tf.float32)
drop = tf.nn.dropout(l7, keep_prob)

w8 = tf.Variable(weight_variable([2000, num_labels]))
b8 = tf.Variable(bias_variable([num_labels]))
lastLayer = (tf.matmul(drop, w8) + b8)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer,tfy))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
prediction=tf.nn.softmax(lastLayer)

model_saver = tf.train.Saver({  'w1': w1, 'b1': b1, 'beta1': beta1, 'scale1': scale1,
                                'w2': w2, 'b2': b2, 'beta2': beta2, 'scale2': scale2,
                                'w3': w3, 'b3': b3, 'beta3': beta3, 'scale3': scale3,
                                'w4': w4, 'b4': b4, 'beta4': beta4, 'scale4': scale4,
                                'w5': w5, 'b5': b5, 'beta5': beta5, 'scale5': scale5,
                                'w6': w6, 'b6': b6, 'beta6': beta6, 'scale6': scale6,
                                'w7': w7, 'b7': b7, 'w8': w8, 'b8': b8
                            })

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()

sess.run(init_op)
for i in range(noOfIterations):
    indices = np.random.permutation(X_train.shape[0])[:batch_size]
    crop1 = np.random.randint(0,8)
    crop2 = np.random.randint(0,8)
    X_batch = X_train[indices,crop1:crop1+24,crop2:crop2+24,:]
    y_batch = y_train[indices,:]
    if random.random() < .5:
        X_batch = X_batch[:,:,::-1,:]
    feed_dict = {tfx : X_batch, tfy : y_batch, keep_prob: 0.5}
    _, l, predictions = sess.run([optimizer, loss, prediction], feed_dict=feed_dict)

    if (i % 100 == 0):
        test_accuracies = []
        for j in range(0,X_test.shape[0],batch_size):
            feed_dict={tfx: X_test[j:j+batch_size,4:28,4:28,:],tfy: y_test[j:j+batch_size,:], keep_prob: 1.0}
            l, predictions = sess.run([loss, prediction], feed_dict=feed_dict)
            test_accuracies.append(accuracy(predictions,y_test[j:j+batch_size,:]))
        print 'iteration %i test accuracy: %.1f%%' % (i, np.mean(test_accuracies))
        
    if (i % 5000 == 0):
        print "Saving the model"
        model_saver.save(sess, 'model.ckpt')
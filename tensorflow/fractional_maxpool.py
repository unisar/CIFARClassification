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

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    return initial

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return initial

def conv2d(x, W, stride=[1,1,1,1],pad='VALID'):
    return tf.nn.conv2d(x, W, strides=stride, padding=pad)
	
def fmaxpool(x, ratio=[1.0, 1.25, 1.25, 1.0]):
    h_pool, rows, cols = tf.nn.fractional_max_pool(x,ratio,True,True)
    return h_pool
    
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

tfx = tf.placeholder(tf.float32, shape=[None,image_size,image_size,num_channels])
tfy = tf.placeholder(tf.float32, shape=[None,num_labels])

w1 = tf.Variable(weight_variable([2,2,3,100]))
b1 = tf.Variable(bias_variable([100]))
l1 = tf.nn.elu(conv2d(tfx,w1) + b1)
mp1 = fmaxpool(l1)

w2 = tf.Variable(weight_variable([2,2,100,133]))
b2 = tf.Variable(bias_variable([133]))
l2 = tf.nn.elu(conv2d(mp1, w2) + b2)
mp2 = fmaxpool(l2)

w3 = tf.Variable(weight_variable([2,2,133,178]))
b3 =  tf.Variable(bias_variable([178]))
l3 = tf.nn.elu(conv2d(mp2, w3) + b3)
mp3 = fmaxpool(l3)

w4 = tf.Variable(weight_variable([2,2,178,246]))
b4 =  tf.Variable(bias_variable([246]))
l4 = tf.nn.elu(conv2d(mp3, w4) + b4)
mp4 = fmaxpool(l4)

w5 = tf.Variable(weight_variable([2,2,246,356]))
b5 =  tf.Variable(bias_variable([356]))
l5 = tf.nn.elu(conv2d(mp4, w5) + b5)
mp5 = fmaxpool(l5)

w6 = tf.Variable(weight_variable([2,2,356,533]))
b6 =  tf.Variable(bias_variable([533]))
l6 = tf.nn.elu(conv2d(mp5, w6) + b6)
mp6 = fmaxpool(l6)

w7 = tf.Variable(weight_variable([2,2,533,800]))
b7 =  tf.Variable(bias_variable([800]))
l7 = tf.nn.elu(conv2d(mp6, w7) + b7)
mp7 = fmaxpool(l7)

w8 = tf.Variable(weight_variable([2,2,800,1600]))
b8 =  tf.Variable(bias_variable([1600]))
l8 = tf.nn.elu(conv2d(mp7, w8) + b8)

w9 = tf.Variable(weight_variable([1,1,1600,1600]))
b9 = tf.Variable(bias_variable([1600]))
l9 = tf.nn.elu(conv2d(l8, w9) + b9)

reshape = tf.reshape(l9, [-1, 1600])
w10 = tf.Variable(weight_variable([1600, num_labels]))
b10 = tf.Variable(bias_variable([num_labels]))
lastLayer = tf.matmul(reshape, w10) + b10

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer,tfy))
optimizer = tf.train.AdamOptimizer(0.00002).minimize(loss)
prediction = tf.nn.softmax(lastLayer)

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
        X_batch = X_batch[:,:,::-1,:]
    feed_dict = {tfx:X_batch,tfy:y_batch}
    l,_ = sess.run([loss,optimizer], feed_dict=feed_dict)
    print 'iteration %i loss: %.4f' % (i, l)
    if ((i+1) % 100 == 0):
        with open("accuracy.txt", "a") as f:
            f.write('iteration %i loss: %.4f\n' % (i, l))

    if ((i+1) % 10000 == 0):
        test_accuracies = []
        for j in range(0,X_test.shape[0],batch_size):
            model_average = []
            for p in range(10):
                feed_dict={tfx:X_test[j:j+batch_size,:,:,:],tfy:y_test[j:j+batch_size,:]}
                predictions = sess.run(prediction, feed_dict=feed_dict)
                model_average.append(predictions)
            model_average = np.mean(model_average,0)
            test_accuracies.append(accuracy(predictions,y_test[j:j+batch_size,:]))
        print 'iteration %i test accuracy: %.4f%%' % (i, np.mean(test_accuracies))
        with open("accuracy.txt", "a") as f:
            f.write('iteration %i test accuracy: %.4f%%\n' % (i, np.mean(test_accuracies)))
        preds = []
        for j in range(0,X_full.shape[0],batch_size):
            model_average = []
            for p in range(10):
                feed_dict={tfx:X_full[j:j+batch_size,:,:,:]}
                p = sess.run(prediction, feed_dict=feed_dict)
                model_average.append(p)
            model_average = np.mean(model_average,0)
            preds.append(np.argmax(model_average, 1))
        pred = np.concatenate(preds)
        np.savetxt('prediction.txt',pred,fmt='%.0f')
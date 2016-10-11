import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import random

#load data
try:
    input = np.load('X_train_zca.npy')
except:
    input = np.load('X_train.npy')  
input = input.transpose(0,2,3,1)
labels = np.genfromtxt('../data/y_train.txt')

X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=0.1, random_state=42, stratify=labels)

def one_hot(y):
    retVal = np.zeros((len(y), 10))
    retVal[np.arange(len(y)), y.astype(int)] = 1
    return retVal

y_train = one_hot(y_train)
y_test = one_hot(y_test)

try:
    X_full = np.load('X_test_zca.npy')
except:
    X_full = np.load('X_test.npy')
X_full = X_full.transpose(0,2,3,1)

#model parameters
noOfIterations = 100000
image_size = 32
num_channels = 3
num_labels = 10
batch_size = 100

#layer initialization functions
def conv_ortho_weights(chan_in,filter_h,filter_w,chan_out):
    bound = np.sqrt(6./(chan_in*filter_h*filter_w + chan_out*filter_h*filter_w))
    W = np.random.random((chan_out, chan_in * filter_h * filter_w))
    u, s, v = np.linalg.svd(W,full_matrices=False)
    if u.shape[0] != u.shape[1]:
        W = u.reshape((chan_in, filter_h, filter_w, chan_out))
    else:
        W = v.reshape((chan_in, filter_h, filter_w, chan_out))
    return W.astype(np.float32)

def dense_ortho_weights(fan_in,fan_out):
    bound = np.sqrt(2./(fan_in+fan_out))
    W = np.random.randn(fan_in,fan_out)*bound
    u, s, v = np.linalg.svd(W,full_matrices=False)
    if u.shape[0] != u.shape[1]:
        W = u
    else:
        W = v
    return W.astype(np.float32)
    
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return initial

def conv2d(x, W, stride=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#neural net architecture
tfx = tf.placeholder(tf.float32, shape=[None,image_size,image_size,num_channels])
tfy = tf.placeholder(tf.float32, shape=[None,num_labels])
kp1 = tf.placeholder(tf.float32)
d1 = tf.nn.dropout(tfx, kp1)

w1 = tf.Variable(conv_ortho_weights(3,3,3,80))
b1 = tf.Variable(bias_variable([80]))
l1 = tf.nn.elu(conv2d(d1,w1) + b1)

w2 = tf.Variable(conv_ortho_weights(3,3,80,80))
b2 = tf.Variable(bias_variable([80]))    
l2 = tf.nn.elu(conv2d(l1, w2) + b2)
maxpool1 = max_pool_2x2(l2)

w3 = tf.Variable(conv_ortho_weights(3,3,80,160))
b3 = tf.Variable(bias_variable([160]))
l3 = tf.nn.elu(conv2d(maxpool1, w3) + b3)

w4 = tf.Variable(conv_ortho_weights(3,3,160,160))
b4 =  tf.Variable(bias_variable([160]))
l4 = tf.nn.elu(conv2d(l3, w4) + b4)
maxpool2 = max_pool_2x2(l4)

w5 = tf.Variable(conv_ortho_weights(3,3,160,320))
b5 =  tf.Variable(bias_variable([320]))
l5 = tf.nn.elu(conv2d(maxpool2, w5) + b5)

w6 = tf.Variable(conv_ortho_weights(3,3,320,320))
b6 =  tf.Variable(bias_variable([320]))
l6 = tf.nn.elu(conv2d(l5, w6) + b6)

w7 = tf.Variable(dense_ortho_weights(8 * 8 * 320, 2000))
b7 = tf.Variable(bias_variable([2000]))
flattened = tf.reshape(l6, [-1, 8 * 8 * 320])
l7 = tf.nn.elu(tf.matmul(flattened, w7) + b7)

kp2 = tf.placeholder(tf.float32)
drop = tf.nn.dropout(l7, kp2)

w8 = tf.Variable(dense_ortho_weights(2000, num_labels))
b8 = tf.Variable(bias_variable([num_labels]))
lastLayer = tf.matmul(drop, w8) + b8

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer,tfy))
optimizer = tf.train.AdamOptimizer(0.0002,0.9,0.99).minimize(loss)
prediction=tf.nn.softmax(lastLayer)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(tfy,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run model
init_op = tf.initialize_all_variables()
sess = tf.InteractiveSession()
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
    feed_dict = {tfx:X_batch,tfy:y_batch,kp1:0.8,kp2:0.5}
    l,_ = sess.run([loss,optimizer], feed_dict=feed_dict)
    print 'iteration %i loss: %.4f' % (i, l)

    if (i % 100 == 0):
        test_accuracies = []
        for j in range(0,X_test.shape[0],batch_size):
            feed_dict={tfx:X_test[j:j+batch_size,:,:,:],tfy:y_test[j:j+batch_size,:],kp1:1.,kp2:1.}
            test_accuracies.append(sess.run(accuracy, feed_dict=feed_dict)*100)
        print 'iteration %i test accuracy: %.4f%%' % (i, np.mean(test_accuracies))
        with open("accuracy.txt", "a") as f:
            f.write('iteration %i test accuracy: %.4f%%\n' % (i, np.mean(test_accuracies)))
            f.write('iteration %i training loss: %.4f\n' % (i, l))
 
    if (i % 5000 == 0):
        preds = []
        for j in range(0,X_full.shape[0],batch_size):
            feed_dict={tfx:X_full[j:j+batch_size,:,:,:],kp1:1.,kp2:1.}
            p = sess.run(prediction, feed_dict=feed_dict)
            preds.append(np.argmax(p, 1))
        pred = np.concatenate(preds)
        np.savetxt('prediction.txt',pred,fmt='%.0f')
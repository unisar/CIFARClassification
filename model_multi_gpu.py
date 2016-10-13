import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import random
import time

#load training data
try:
    input = np.load('X_train_zca.npy')
except:
    input = np.load('X_train.npy')  
input = input.transpose(0,2,3,1)
labels = np.genfromtxt('../data/y_train.txt')

#cross validation splitting
X_train, X_test, y_train, y_test = train_test_split(input, labels, test_size=0.1, random_state=42, stratify=labels)

#one hot encoding for labels
def one_hot(y):
    retVal = np.zeros((len(y), 10))
    retVal[np.arange(len(y)), y.astype(int)] = 1
    return retVal

y_train = one_hot(y_train)
y_test = one_hot(y_test)

#load test data
try:
    X_full = np.load('X_test_zca.npy')
except:
    X_full = np.load('X_test.npy')
X_full = X_full.transpose(0,2,3,1)

#model parameters
noOfIterations = 80000
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
kp2 = tf.placeholder(tf.float32)
w1 = tf.Variable(conv_ortho_weights(3,3,3,80),name='w1')
b1 = tf.Variable(bias_variable([80]),name='b1')
w2 = tf.Variable(conv_ortho_weights(3,3,80,80),name='w2')
b2 = tf.Variable(bias_variable([80]),name='b2')
w3 = tf.Variable(conv_ortho_weights(3,3,80,160),name='w3')
b3 = tf.Variable(bias_variable([160]),name='b3')
w4 = tf.Variable(conv_ortho_weights(3,3,160,160),name='w4')
b4 =  tf.Variable(bias_variable([160]),name='b4')
w5 = tf.Variable(conv_ortho_weights(3,3,160,320),name='w5')
b5 =  tf.Variable(bias_variable([320]),name='b5')
w6 = tf.Variable(conv_ortho_weights(3,3,320,320),name='w6')
b6 =  tf.Variable(bias_variable([320]),name='b6')
w7 = tf.Variable(dense_ortho_weights(8 * 8 * 320, 2000),name='w7')
b7 = tf.Variable(bias_variable([2000]),name='b7')
w8 = tf.Variable(dense_ortho_weights(2000, num_labels),name='w8')
b8 = tf.Variable(bias_variable([num_labels]),name='b8')
optimizer = tf.train.AdamOptimizer(0.0002,0.9,0.99)

with tf.device('/gpu:0'):
    tfx_1 = tf.slice(tfx,[0,0,0,0],[batch_size/4,32,32,3])
    tfy_1 = tf.slice(tfy,[0,0],[batch_size/4,10])
    d1_1 = tf.nn.dropout(tfx_1, kp1)
    l1_1 = tf.nn.elu(conv2d(d1_1,w1) + b1)
    l2_1 = tf.nn.elu(conv2d(l1_1, w2) + b2)
    maxpool1_1 = max_pool_2x2(l2_1)
    l3_1 = tf.nn.elu(conv2d(maxpool1_1, w3) + b3)
    l4_1 = tf.nn.elu(conv2d(l3_1, w4) + b4)
    maxpool2_1 = max_pool_2x2(l4_1)
    l5_1 = tf.nn.elu(conv2d(maxpool2_1, w5) + b5)
    l6_1 = tf.nn.elu(conv2d(l5_1, w6) + b6)
    flattened_1 = tf.reshape(l6_1, [-1, 8 * 8 * 320])
    l7_1 = tf.nn.elu(tf.matmul(flattened_1, w7) + b7)
    drop_1 = tf.nn.dropout(l7_1, kp2)
    lastLayer_1 = tf.matmul(drop_1, w8) + b8
    loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer_1,tfy_1))
    grad_1 = optimizer.compute_gradients(loss_1)

with tf.device('/gpu:1'):
    tfx_2 = tf.slice(tfx,[batch_size/4,0,0,0],[batch_size/4,32,32,3])
    tfy_2 = tf.slice(tfy,[batch_size/4,0],[batch_size/4,10])
    d1_2 = tf.nn.dropout(tfx_2, kp1)
    l1_2 = tf.nn.elu(conv2d(d1_2,w1) + b1)
    l2_2 = tf.nn.elu(conv2d(l1_2, w2) + b2)
    maxpool1_2 = max_pool_2x2(l2_2)
    l3_2 = tf.nn.elu(conv2d(maxpool1_2, w3) + b3)
    l4_2 = tf.nn.elu(conv2d(l3_2, w4) + b4)
    maxpool2_2 = max_pool_2x2(l4_2)
    l5_2 = tf.nn.elu(conv2d(maxpool2_2, w5) + b5)
    l6_2 = tf.nn.elu(conv2d(l5_2, w6) + b6)
    flattened_2 = tf.reshape(l6_2, [-1, 8 * 8 * 320])
    l7_2 = tf.nn.elu(tf.matmul(flattened_2, w7) + b7)
    drop_2 = tf.nn.dropout(l7_2, kp2)
    lastLayer_2 = tf.matmul(drop_2, w8) + b8
    loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer_2,tfy_2))
    grad_2 = optimizer.compute_gradients(loss_2)
    
with tf.device('/gpu:2'):
    tfx_3 = tf.slice(tfx,[2*batch_size/4,0,0,0],[batch_size/4,32,32,3])
    tfy_3 = tf.slice(tfy,[2*batch_size/4,0],[batch_size/4,10])
    d1_3 = tf.nn.dropout(tfx_3, kp1)
    l1_3 = tf.nn.elu(conv2d(d1_3,w1) + b1)
    l2_3 = tf.nn.elu(conv2d(l1_3, w2) + b2)
    maxpool1_3 = max_pool_2x2(l2_3)
    l3_3 = tf.nn.elu(conv2d(maxpool1_3, w3) + b3)
    l4_3 = tf.nn.elu(conv2d(l3_3, w4) + b4)
    maxpool2_3 = max_pool_2x2(l4_3)
    l5_3 = tf.nn.elu(conv2d(maxpool2_2, w5) + b5)
    l6_3 = tf.nn.elu(conv2d(l5_3, w6) + b6)
    flattened_3 = tf.reshape(l6_3, [-1, 8 * 8 * 320])
    l7_3 = tf.nn.elu(tf.matmul(flattened_3, w7) + b7)
    drop_3 = tf.nn.dropout(l7_3, kp2)
    lastLayer_3 = tf.matmul(drop_3, w8) + b8
    loss_3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer_3,tfy_3))
    grad_3 = optimizer.compute_gradients(loss_3)
    
with tf.device('/gpu:3'):
    tfx_4 = tf.slice(tfx,[3*batch_size/4,0,0,0],[batch_size/4,32,32,3])
    tfy_4 = tf.slice(tfy,[3*batch_size/4,0],[batch_size/4,10])
    d1_4 = tf.nn.dropout(tfx_4, kp1)
    l1_4 = tf.nn.elu(conv2d(d1_4,w1) + b1)
    l2_4 = tf.nn.elu(conv2d(l1_4, w2) + b2)
    maxpool1_4 = max_pool_2x2(l2_4)
    l3_4 = tf.nn.elu(conv2d(maxpool1_4, w3) + b3)
    l4_4 = tf.nn.elu(conv2d(l3_4, w4) + b4)
    maxpool2_4 = max_pool_2x2(l4_4)
    l5_4 = tf.nn.elu(conv2d(maxpool2_2, w5) + b5)
    l6_4 = tf.nn.elu(conv2d(l5_4, w6) + b6)
    flattened_4 = tf.reshape(l6_4, [-1, 8 * 8 * 320])
    l7_4 = tf.nn.elu(tf.matmul(flattened_4, w7) + b7)
    drop_4 = tf.nn.dropout(l7_4, kp2)
    lastLayer_4 = tf.matmul(drop_4, w8) + b8
    loss_4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer_4,tfy_4))
    grad_4 = optimizer.compute_gradients(loss_4)

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
    
grads = average_gradients([grad_1,grad_2,grad_3,grad_4])
loss = tf.reduce_mean([loss_1,loss_2,loss_3,loss_4], 0)
apply_gradient = optimizer.apply_gradients(grads)
lastLayer = tf.concat(0, [lastLayer_1,lastLayer_2,lastLayer_3,lastLayer_4])
prediction=tf.nn.softmax(lastLayer)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(tfy,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#run model
init_op = tf.initialize_all_variables()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init_op)
open('accuracy.txt', 'w').close()

for i in range(noOfIterations):
    start = time.time()
    indices = np.random.permutation(X_train.shape[0])[:batch_size]
    
    #generate random minibatch
    X_batch = X_train[indices,:,:,:]
    
    #random translation of image by 5 image
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
    
    #randomly flip image
    if random.random() < .5:
        X_batch = X_batch[:,:,::-1,:]
    
    #train
    feed_dict = {tfx:X_batch,tfy:y_batch,kp1:0.8,kp2:0.5}
    l,_ = sess.run([loss,apply_gradient], feed_dict=feed_dict)
    end = time.time()
    print 'time elapsed: ', end - start
    print 'iteration %i loss: %.4f' % (i, l)
    
    #cross validation accuracy
    if (i % 100 == 0):
        test_accuracies = []
        for j in range(0,X_test.shape[0],batch_size):
            feed_dict={tfx:X_test[j:j+batch_size,:,:,:],tfy:y_test[j:j+batch_size,:],kp1:1.,kp2:1.}
            test_accuracies.append(sess.run(accuracy, feed_dict=feed_dict)*100)
        print 'iteration %i test accuracy: %.4f%%' % (i, np.mean(test_accuracies))
        with open("accuracy.txt", "a") as f:
            f.write('iteration %i test accuracy: %.4f%%\n' % (i, np.mean(test_accuracies)))
            f.write('iteration %i training loss: %.4f\n' % (i, l))
    
    #run model on test
    if (i % 5000 == 0):
        preds = []
        for j in range(0,X_full.shape[0],batch_size):
            feed_dict={tfx:X_full[j:j+batch_size,:,:,:],kp1:1.,kp2:1.}
            p = sess.run(prediction, feed_dict=feed_dict)
            preds.append(np.argmax(p, 1))
        pred = np.concatenate(preds)
        np.savetxt('prediction.txt',pred,fmt='%.0f')
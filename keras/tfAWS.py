import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split
import sys


def one_hot(y):
    retVal = np.zeros((len(y), 10))
    retVal[np.arange(len(y)), y.astype(int)] = 1
    return retVal


#Read Input
args = (sys.argv)

#input must be of shape: <no_of_inputs,w,h,num_channels>
X_ = np.load(args[1])   
y_ = np.genfromtxt(args[2])
noOfIterations = int(args[3])

y_ = one_hot(y_)
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.1, random_state=42)


image_size = 32
num_channels = 3
num_labels=10
batch_size = 128
patch_size = 3

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return initial

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
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

w1 = tf.Variable(weight_variable([3,3,3,16]))    
b1 = tf.Variable(bias_variable([16]))
l1 = tf.nn.relu(conv2d(tfx,w1) + b1)


w2 = tf.Variable(weight_variable([3,3,16,32]))
b2 = tf.Variable(bias_variable([32]))    
l2 = tf.nn.relu(conv2d(l1, w2) + b2)
maxpool1 = max_pool_2x2(l2, [1, 2, 2, 1])

w3 = tf.Variable(weight_variable([3,3,32,64]))
b3 = tf.Variable(bias_variable([64]))
l3 = tf.nn.relu(conv2d(maxpool1, w3) + b3)

w4 = tf.Variable(weight_variable([3,3,64,64]))
b4 =  tf.Variable(bias_variable([64]))
l4 = tf.nn.relu(conv2d(l3, w4) + b4)
maxpool2 = max_pool_2x2(l4, [1, 2, 2, 1])

w7 = tf.Variable(weight_variable([3,3,64,128]))
b7 =  tf.Variable(bias_variable([128]))
l7 = tf.nn.relu(conv2d(maxpool2, w7) + b7)

shape = l7.get_shape().as_list()
w5 = tf.Variable(weight_variable([8 * 8 * 128, 128]))
b5 = tf.Variable(bias_variable([128]))
reshape = tf.reshape(l7, [-1, shape[1] * shape[2] * shape[3]])
l5 = tf.nn.relu(tf.matmul(reshape, w5) + b5)
drop = tf.nn.dropout(l5,1.0)


w6 = tf.Variable(weight_variable([128, num_labels]))
b6 = tf.Variable(bias_variable([num_labels]))
lastLayer = (tf.matmul(drop, w6) + b6)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer,tfy))
#learning_rate = tf.Variable(0.01)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
prediction=tf.nn.softmax(lastLayer)

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()


model_saver = tf.train.Saver({  'w1': w1, 'b1': b1, 
                                'w2': w2, 'b2': b2, 
                                'w3': w3, 'b3': b3, 
                                'w4': w4, 'b4': b4,
                                'w7': w7, 'b7': b7,
                                'w5': w5, 'b5': b5,
                                'w6': w6, 'b6': b6
                            })


sess.run(init_op)
for i in range(noOfIterations):
    indices = np.random.permutation(X_.shape[0])[:batch_size]
    X_batch = X_[indices,:,:,:]
    y_batch = y_[indices,:]
    feed_dict = {tfx : X_batch, tfy : y_batch}
    _, l, predictions = sess.run([optimizer, loss, prediction], feed_dict=feed_dict)

    if (i % 50 == 0):
        print("Iteration: %i. Train loss %.5f, Minibatch accuracy: %.1f%%" % (i,l,accuracy(predictions,y_batch)))


feed_dict={tfx: X_test,tfy: y_test}
l, predictions = sess.run([loss, prediction], feed_dict=feed_dict)
print('Test accuracy: %.1f%%' % accuracy(predictions,y_test))

print ("Saving the model")
model_saver.save(sess, 'model.ckpt')
print ("Model Saved")



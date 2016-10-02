import tensorflow as tf
import numpy as np
from sklearn.cross_validation import train_test_split


def one_hot(y):
    retVal = np.zeros((len(y), 10))
    retVal[np.arange(len(y)), y.astype(int)] = 1
    return retVal

X_train = np.load('./npys/X_large_train_subset_10000.npy')   
y_train = np.genfromtxt('./data/y_large_train_subset_10000.txt')
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

y_train = one_hot(y_train)
y_test = one_hot(y_test)


image_size = 32
num_channels = 3
num_labels=10
batch_size = 150
patch_size = 3

sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x,stride):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=stride, padding='SAME')

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


tfx = tf.placeholder(tf.float32, shape=[None,image_size,image_size,num_channels])
tfy = tf.placeholder(tf.float32, shape=[None,num_labels])


epsilon = 1e-3
conv = conv2d(tfx,weight_variable([3,3,3,16]))

#Batch Normalization
batch_mean2, batch_var2 = tf.nn.moments(conv,axes=[0,1,2],keep_dims=False)
scale2 = tf.Variable(tf.ones([16]))
beta2 = tf.Variable(tf.zeros([16]))
BN2 = tf.nn.batch_normalization(conv,batch_mean2,batch_var2,beta2,scale2,epsilon)
hidden = tf.nn.relu(BN2 + bias_variable([16]))


conv = conv2d(hidden, weight_variable([3,3,16,32]))
hidden = tf.nn.relu(conv + bias_variable([32]))
#maxpool = max_pool_2x2(hidden, [1, 2, 2, 1])


conv = conv2d(hidden, weight_variable([3,3,32,64]))
hidden = tf.nn.relu(conv + bias_variable([64]))
#maxpool = hidden
#maxpool = max_pool_2x2(hidden, [1, 2, 2, 1])

conv = conv2d(hidden, weight_variable([3,3,64,64]))
hidden = tf.nn.relu(conv + bias_variable([64]))
#maxpool = max_pool_2x2(hidden, [1, 2, 2, 1])
maxpool = hidden


shape = maxpool.get_shape().as_list()
layer3_weights = tf.Variable(tf.truncated_normal([32 * 32 * 64, 64], stddev=0.1))
layer3_biases = tf.Variable(tf.constant(1.0, shape=[64]))
reshape = tf.reshape(maxpool, [-1, shape[1] * shape[2] * shape[3]])
hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
drop = tf.nn.dropout(hidden, 1.0)

layer4_weights = tf.Variable(tf.truncated_normal([64, num_labels], stddev=0.1))
layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
lastLayer = tf.nn.relu(tf.matmul(drop, layer4_weights) + layer4_biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lastLayer,tfy))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
train_prediction=tf.nn.softmax(lastLayer)


tf.initialize_all_variables().run()

for i in range(2000):
    indices = np.random.permutation(X_train.shape[0])[:batch_size]
    X_batch = X_train[indices,:,:,:]
    y_batch = y_train[indices,:]
    feed_dict = {tfx : X_batch, tfy : y_batch}
    _, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

    if (i % 50 == 0):
        # Test trained model
        #correct_prediction = tf.equal(tf.argmax(predictions, 1),tf.argmax(tfy, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #print('Minibatch loss at step %d: %f' % (i, l))
        print('Minibatch accuracy: %.1f%%' % accuracy(predictions,y_batch))


feed_dict={tfx: X_test,tfy: y_test}
_, l, predictions = sess.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
print('Test accuracy: %.1f%%' % accuracy(predictions,y_test))
#print("test acc: %i " % acc)



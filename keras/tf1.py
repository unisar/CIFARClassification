import tensorflow as tf
import numpy as np


def reformat(dataset, labels):
  dataset = dataset.reshape((-1, 32, 32, 3)).astype(np.float32)
  #dataset = dataset.astype(np.float32)
  labels = (np.arange(10) == labels[:,None]).astype(np.float32)
  return dataset, labels

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


X_train = np.load('./X_small_train_1.npy')   
y_train = np.genfromtxt('../data/y_small_train.txt')
X_train, y_train = reformat(X_train, y_train)
print (y_train.shape)

X_test = np.load('./X_small_test_1.npy')   
y_test = np.genfromtxt('../data/y_small_test.txt')
X_test, y_test = reformat(X_test, y_test)
print ("Train and test read!!")

image_size = 32
num_channels = 3
num_labels=10
batch_size = 16
patch_size = 3
depth = 16
num_hidden = 64

graph = tf.Graph()
with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size,num_labels))
  tf_test_dataset = tf.constant(X_test)
  tf_test_labels = tf.constant(y_test)

  def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  layer3_weights = tf.Variable(tf.truncated_normal([image_size * image_size  * 128, 128], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[128]))
  layer4_weights = tf.Variable(tf.truncated_normal([128, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  # Model.
  def model(data):
        #(3,3,3,16)
        conv = tf.nn.conv2d(data, weight_variable([3,3,3,16]), [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + bias_variable([16]))
        maxpool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        #(3,3,16,16)
        conv = tf.nn.conv2d(maxpool, weight_variable([3,3,16,16]), [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + bias_variable([16]))
        maxpool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        #(3,3,16,32)
        conv = tf.nn.conv2d(maxpool, weight_variable([3,3,16,32]), [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + bias_variable([32]))
        maxpool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        #(3,3,32,32)
        conv = tf.nn.conv2d(maxpool, weight_variable([3,3,32,32]), [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + bias_variable([32]))
        maxpool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        #(3,3,32,64)
        conv = tf.nn.conv2d(maxpool, weight_variable([3,3,32,64]), [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + bias_variable([64]))
        maxpool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        #(3,3,64,64)
        conv = tf.nn.conv2d(maxpool, weight_variable([3,3,64,64]), [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + bias_variable([64]))
        maxpool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
        
        #(3,3,64,128)
        conv = tf.nn.conv2d(maxpool, weight_variable([3,3,64,128]), [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + bias_variable([128]))
        maxpool = tf.nn.max_pool(hidden, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        shape = maxpool.get_shape().as_list()
        reshape = tf.reshape(maxpool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

        shape = reshape.get_shape().as_list()
        reshape = tf.reshape(reshape, [shape[0], shape[1]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases
        

  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
      
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  test_prediction = tf.nn.softmax(model(tf_test_dataset)) 
  num_steps = 1000

  with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        for step in range(num_steps):
            offset = (step * batch_size) % (y_train.shape[0] - batch_size)
            batch_data = X_train[offset:(offset + batch_size), :, :, :]
            batch_labels = y_train[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 50 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), y_test))   


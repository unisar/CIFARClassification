
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import multiGPU
np.set_printoptions(threshold=np.nan)


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/home/ubuntu/cifar10_eval',"""Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',"""Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ubuntu/cifar10_train',"""Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 15000,"""Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,"""Whether to run eval only once.""")
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.


sess = tf.InteractiveSession()

#Read Input
X_train = np.load('X_train_large_subset_15001_30000.npy')   

def one_hot(y):
    retVal = np.zeros((len(y), 10))
    retVal[np.arange(len(y)), y.astype(int)] = 1
    return retVal

images = tf.placeholder(tf.float32, shape=[None,32,32,3])


# Randomly crop a [height, width] section of the image.
#distorted_images =  tf.map_fn(lambda img: tf.random_crop(img, [32, 32, 3]), images)
#tf.random_crop(reshaped_image, [height, width, 3])

# Randomly flip the image horizontally.
distorted_images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
      
# Because these operations are not commutative, consider randomizing
# the order their operation.
distorted_images = tf.map_fn(lambda img: tf.image.random_brightness(img,max_delta=63), distorted_images)

distorted_images = tf.map_fn(lambda img: tf.image.random_contrast(img,lower=0.2, upper=1.8), distorted_images)

# Subtract off the mean and divide by the variance of the pixels.
float_images = tf.map_fn(lambda img: tf.image.per_image_whitening(img), distorted_images)


# Build a Graph that computes the logits predictions from the
# inference model.
logits = multiGPU.inference(float_images)
    
# Calculate predictions.
prediction=tf.nn.softmax(logits)

variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

#restorer = tf.train.Saver(tf.all_variables())
ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
print("ckpt Directory found!!")

saver.restore(sess, ckpt.model_checkpoint_path)
preds = []
for j in range(0,X_train.shape[0],100):
    feed_dict={images:X_train[j:j+100,:,:,:]}
    p = sess.run(prediction, feed_dict=feed_dict)
    preds.append(np.argmax(p, 1))

pred = np.concatenate(preds)
np.savetxt('prediction.txt',pred,fmt='%.0f')

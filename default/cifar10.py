from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 200.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def training_inputs():
  images, labels = cifar10_input.training_inputs(batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inference(images, kp0=0.8, kp1=0.5, kp2=0.5, batchSize=FLAGS.batch_size):
  print ('kp0 %.f, kp1 %.f, kp2 %.f' % (kp0, kp1, kp2))
  # drop1
  with tf.variable_scope('drop1') as scope:
    drop1 = tf.nn.dropout(images, keep_prob=kp0, name='drop1')

  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 3, 96],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(drop1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv1)

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 96, 96],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv2)

  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 96, 96],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv3)

  # drop2
  with tf.variable_scope('drop2') as scope:
    drop2 = tf.nn.dropout(conv3, keep_prob=kp1, name='drop2')

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 96, 192],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(drop2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv4)

  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 192, 192],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv5)

  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 192, 192],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv5, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv6)

  # drop3
  with tf.variable_scope('drop3') as scope:
    drop3 = tf.nn.dropout(conv6, keep_prob=kp2, name='drop3')

  # conv7
  with tf.variable_scope('conv7') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[3, 3, 192, 192],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(drop3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv7 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv7)

  # conv8
  with tf.variable_scope('conv8') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1, 192, 192],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv7, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv8 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv8)

  # conv9
  with tf.variable_scope('conv9') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1, 192, 10],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(conv8, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv9 = tf.nn.relu(bias, name=scope.name)
    _activation_summary(conv9)

  # avg
  with tf.variable_scope('avg') as scope:
    avg = tf.nn.avg_pool(conv9, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1],
                         padding='VALID', name='avg')
  
  with tf.variable_scope('lastlayer') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(avg, [batchSize, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('softmax', shape=[10, 10],stddev=0.05,wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(reshape, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels):
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

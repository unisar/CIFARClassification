from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import dtypes

IMAGE_SIZE = 32

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

def read_cifar10(filename_queue):
  class CIFAR10Record(object):
    pass
  result = CIFAR10Record()
  label_bytes = 1 
  result.height = 32
  result.width = 32
  result.depth = 3
  image_bytes = result.height * result.width * result.depth
  record_bytes = label_bytes + image_bytes
  value = tf.read_file(filename_queue[0])
  result.uint8image = tf.image.decode_png(value, 3)
  result.label = filename_queue[1]
  return result

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size):
  num_preprocess_threads = 32
  images, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples + 3 * batch_size)

  tf.image_summary('images', images)
  return images, tf.reshape(label_batch, [batch_size])

def read_integers(filename):
    with open(filename) as f:
        return [int(x) for x in f]

def training_inputs(batch_size):
  alllabels = read_integers('/labels.txt')
 
  trainLabels = alllabels[:45000]
  print('Length of training labels: ',  len(trainLabels))

  with open('/filenames.txt') as f:
    filenames = f.read().splitlines() # .readlines()

  for i in range(len(filenames)):
    filenames[i] = '/Images/' + filenames[i] + '.png'

  trainFiles = filenames[:45000]

  trainimagesT = tf.convert_to_tensor(trainFiles, dtype=dtypes.string)
  trainlabelsT = tf.convert_to_tensor(trainLabels, dtype=dtypes.int32)

  # Create a queue that produces the filenames to read.  
  filename_queue = tf.train.slice_input_producer([trainimagesT, trainlabelsT])

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
  distorted_image = tf.image.random_flip_left_right(distorted_image)
  distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_whitening(distorted_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size)


def testinginputs(bs):
  num_examples_per_epoch = 5000
  
  alllabels = read_integers('/labels.txt')

  testLabels = alllabels[45000:]

  with open('/filenames.txt') as f:
    filenames = f.read().splitlines() # .readlines()

  for i in range(len(filenames)):
    filenames[i] = '/Images/' + filenames[i] + '.png'

  testFiles = filenames[45000:]

  testimagesT = tf.convert_to_tensor(testFiles, dtype=dtypes.string)
  testlabelsT = tf.convert_to_tensor(testLabels, dtype=dtypes.int32)

  filename_queue = tf.train.slice_input_producer([testimagesT, testlabelsT], shuffle=False)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  resized_image = tf.reshape(reshaped_image, [32, 32, 3])
  float_image = tf.image.per_image_whitening(resized_image)

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

  print (float_image.get_shape())
  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, bs)

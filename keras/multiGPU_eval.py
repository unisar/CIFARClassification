# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10
import multiGPU2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/ubuntu/cifar10_eval',"""Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',"""Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ubuntu/cifar10_train',"""Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,"""Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,"""Whether to run eval only once.""")
IMAGE_SIZE = 32
BATCH_SIZE = 100
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
MOVING_AVERAGE_DECAY = 0.09


def eval_once(saver, prediction):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))
    
      num_iter = int(math.ceil(10000/100))
      step = 0
      preds=[]
      while step < num_iter and not coord.should_stop():
          p = sess.run(prediction)
          preds.append(np.argmax(p, 1))
          step = step + 1
          print("Step: %i",step)
      
      pred = np.concatenate(preds)
      np.savetxt('prediction.txt', pred, fmt='%.0f')

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)
      
    coord.request_stop()
    coord.join(threads,stop_grace_period_secs=5)
   
def read_images_from_disk(img_Q):
      fileName = img_Q[0]
      file_contents = tf.read_file(fileName)
      example = tf.image.decode_png(file_contents, channels=3,dtype=tf.uint8)
      example.set_shape([32, 32, 3])
      return example

def read_and_preprocess_images():
      data_dir="/home/ubuntu/mytf/data/X_test.txt"
      filenames = np.genfromtxt(data_dir,dtype="string")
      print ("read the filenames!!")
      
      image_list = []
      for i in xrange(len(filenames)):
        fn = filenames[i].replace('\n','').strip()
        image_list.append("/home/ubuntu/mytf/images/%s.png" % fn)

      images = tf.convert_to_tensor(image_list, dtype=tf.string)
      print ("converted to tensors!")

      # Makes an input queue
      input_queue = tf.train.slice_input_producer([images])
      print ("sliciing done")

      #Read individual image, label from the queue
      image = read_images_from_disk(input_queue)
      image = tf.cast(image, tf.float32)
      print ("read_images_from_disk done")
      
      # Image processing for evaluation.
      # Crop the central [height, width] of the image.
      distorted_image = tf.image.random_flip_left_right(image)

      distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
      
      distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
      
      # Subtract off the mean and divide by the variance of the pixels.
      distorted_image = tf.image.per_image_whitening(distorted_image)

      # Ensure that the random shuffling has good mixing properties.
      min_fraction_of_examples_in_queue = 0.4
      num_preprocess_threads = 4
      min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL * min_fraction_of_examples_in_queue)
      images = tf.train.batch([distorted_image],batch_size=BATCH_SIZE,num_threads=num_preprocess_threads,capacity=min_queue_examples + 3 * BATCH_SIZE)
  
      return images

def evaluate():

  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    #eval_data = FLAGS.eval_data == 'test'
    
    # Get images and labels for CIFAR-10.
    images = read_and_preprocess_images()
    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = multiGPU2.inference(images)

    preds = tf.nn.softmax(logits)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    
    while True:
      eval_once(saver,preds)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  main()

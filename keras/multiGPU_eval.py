from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10
import multiGPU

np.set_printoptions(threshold=np.nan)
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/home/ubuntu/cifar10_eval',"""Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',"""Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ubuntu/cifar10_train',"""Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,"""How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 1000,"""Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,"""Whether to run eval only once.""")



# Constants describing the training process.
TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
IMAGE_SIZE = 32
RUN_ONCE = True
final_predictions = []

def one_hot(y):
    retVal = np.zeros((len(y), 10))
    retVal[np.arange(len(y)), y.astype(int)] = 1
    return retVal

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])


def eval_once(saver, top_k_op, logits):
  print("in eval_once!!")

  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    global final_predictions 
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    print("ckpt Directory found!!")
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
      x = sess.run([logits])
      print (x.eval())
    
      print("in try block")
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / 100))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * 100
      step = 0
      print("step < num_iter")
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1
        
        
      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads)
    

def evaluate():
  X_train = np.load('X_train_large_subset_15001_30000.npy')
  X_train = X_train.astype(np.float32)

  #labels = np.genfromtxt('y_train.txt')
  y_train = np.genfromtxt('y_train_large_subset_15001_30000.txt')
  #y_train = one_hot(y_train)
  
  print("everything readd!!!")
    
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels for CIFAR-10.
    #eval_data = FLAGS.eval_data == 'test'
   
    #READ our IMAGES!!!
    for i in range(11):
      conv_img = tf.convert_to_tensor(X_train[i],dtype=tf.float32)
      conv_label = tf.convert_to_tensor(y_train[i],dtype=tf.int32)
      images, labels = tf.train.batch([conv_img, conv_label],batch_size=100)

    
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = multiGPU.inference(images)
    
    
    # Calculate predictions.
    #labels = tf.argmax(labels, 1)
    #prediction=tf.nn.softmax(logits)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    
    print("top_k_op done!!")

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    print("saver done!!")

    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()

    # summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(saver, top_k_op, logits)
      if RUN_ONCE:
        break
      time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):
    print("in main!!!")
    evaluate()

if __name__ == '__main__':
  main()
  #tf.app.run()
from datetime import datetime
import os.path
import re
import time
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/home/ubuntu/cifar10_train',"""Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 50000, """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 4, """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', True,"""Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")

# Constants describing the training process.
TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 10000      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
IMAGE_SIZE = 32
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 15000
#50000
BATCH_SIZE = 100


def one_hot(y):
    retVal = np.zeros((len(y), 10))
    retVal[np.arange(len(y)), y.astype(int)] = 1
    return retVal

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def inference(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 3, 64],stddev=5e-2,wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope.name)
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
    
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 64, 64],stddev=5e-2,wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 2, 2, 1], padding='VALID')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope.name)
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [BATCH_SIZE,shape[1]*shape[2]*shape[3]])
   
  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [BATCH_SIZE,-1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
  
  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
  
  # softmax, i.e. softmax(WX + b)
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, 10],stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [10],tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
  
  return softmax_linear

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.
  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  Returns:
    Loss tensor of type float.
  """
  
  # Calculate the average cross entropy loss across the batch.
  #labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,labels,name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def tower_loss(scope, images, labels):
  """Calculate the total loss on a single tower running the CIFAR model.
  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Get images and labels for CIFAR-10.
  
  # Build inference Graph.
  logits = inference(images)
  
  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summmary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(loss_name +' (raw)', l)
    tf.scalar_summary(loss_name, loss_averages.average(l))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
    X_train = np.load('X_train_large_subset_1_15000.npy')
    X_train = X_train.astype(np.float32)

    y_train = np.genfromtxt('y_train_large_subset_1_15000.txt')
    y_train = one_hot(y_train)
    
    with tf.Graph().as_default(), tf.device('/cpu:0'):
      # Create a variable to count the number of train() calls. This equals the
      # number of batches processed * FLAGS.num_gpus.
      global_step = tf.get_variable('global_step', [],initializer=tf.constant_initializer(0), trainable=False)

      # Calculate the learning rate schedule.
      num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE)
      decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

      # Decay the learning rate exponentially based on the number of steps.
      lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)

      # Create an optimizer that performs gradient descent.
      opt = tf.train.GradientDescentOptimizer(lr)
     
      #READ our IMAGES!!!
      images = tf.placeholder(tf.float32, shape=[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3])

      # Randomly crop a [height, width] section of the image.
      #distorted_images =  tf.map_fn(lambda img: tf.random_crop(img, [height, width, 3]), images)
      #tf.random_crop(reshaped_image, [height, width, 3])

      # Randomly flip the image horizontally.
      distorted_images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), images)
      
      # Because these operations are not commutative, consider randomizing
      # the order their operation.
      distorted_images = tf.map_fn(lambda img: tf.image.random_brightness(img,max_delta=63), distorted_images)

      distorted_images = tf.map_fn(lambda img: tf.image.random_contrast(img,lower=0.2, upper=1.8), distorted_images)

      # Subtract off the mean and divide by the variance of the pixels.
      float_images = tf.map_fn(lambda img: tf.image.per_image_whitening(img), distorted_images)
      

      labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE,10])
  

      #Calculate the gradients for each model tower.
      tower_grads = []
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
            print('Working on: %s_%d' % (TOWER_NAME, i))
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss = tower_loss(scope,float_images,labels)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

      # We must calculate the mean of each gradient. Note that this is the
      # synchronization point across all towers.
      grads = average_gradients(tower_grads)

      # Apply the gradients to adjust the shared variables.
      apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

      # Track the moving averages of all trainable variables.
      variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
      variables_averages_op = variable_averages.apply(tf.trainable_variables())

      # Group all updates to into a single train op.
      train_op = tf.group(apply_gradient_op, variables_averages_op)

      # Create a saver.
      saver = tf.train.Saver(tf.all_variables())

      # Build an initialization operation to run below.
      init = tf.initialize_all_variables()


      # Calculate predictions.
      #prediction=tf.nn.softmax(logits)


      # Start running operations on the Graph. allow_soft_placement must be set to
      # True to build towers on GPU, as some of the ops do not have GPU
      # implementations.
      sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=FLAGS.log_device_placement))
      sess.run(init)

      
      for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        indices = np.random.permutation(X_train.shape[0])[:100]
        X_batch = X_train[indices,:,:,:]
        y_batch = y_train[indices,:]
        feed_dict = {images : X_batch, labels : y_batch} 
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
          num_examples_per_step = BATCH_SIZE * FLAGS.num_gpus
          examples_per_sec = num_examples_per_step / float(duration)
          sec_per_batch = float(duration) / FLAGS.num_gpus

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ''sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,examples_per_sec, sec_per_batch))

        
        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
  train()


if __name__ == '__main__':
  main()

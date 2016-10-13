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
MOVING_AVERAGE_DECAY = 0.09     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 1000.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.0001  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.
IMAGE_SIZE = 32
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
BATCH_SIZE = 100
NUM_CLASSES = 10

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
  # conv1  
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 3, 96],stddev=5e-2,wd=0.0)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu6(bias, name=scope.name)
    drop1 = tf.nn.dropout(conv1,0.8)
    
  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 96, 96],stddev=5e-2,wd=0.0)
    conv = tf.nn.conv2d(drop1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu6(bias, name=scope.name)
  
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 96, 96],stddev=5e-2,wd=0.0)
    conv = tf.nn.conv2d(conv2, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [96], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu6(bias, name=scope.name)
    drop3 = tf.nn.dropout(conv3,0.5)

  # conv4
  with tf.variable_scope('conv4') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 96, 192],stddev=5e-2,wd=0.0)
    conv = tf.nn.conv2d(drop3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu6(bias, name=scope.name)

  
  # conv5
  with tf.variable_scope('conv5') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 192, 192],stddev=5e-2,wd=0.0)
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu6(bias, name=scope.name)
  
  # conv6
  with tf.variable_scope('conv6') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 192, 192],stddev=5e-2,wd= 0.001)
    conv = tf.nn.conv2d(conv5, kernel, [1, 2, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv6 = tf.nn.relu6(bias)
    drop6 = tf.nn.dropout(conv6,0.5, name=scope.name)


  # conv7
  with tf.variable_scope('conv7') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[3, 3, 192, 192],stddev=5e-2,wd= 0.0)
    conv = tf.nn.conv2d(drop6, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    bias = tf.nn.bias_add(conv, biases)
    conv7 = tf.nn.relu6(bias, name=scope.name)
  
  # conv8
  with tf.variable_scope('conv8') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[1, 1, 192, 192],stddev=5e-2,wd=0.0)
    conv = tf.nn.conv2d(conv7, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv8 = tf.nn.relu6(bias, name=scope.name)

  # conv9
  with tf.variable_scope('conv9') as scope:
    kernel = _variable_with_weight_decay('weights',shape=[1, 1, 192, 10],stddev=0.04,wd=0.0)
    conv = tf.nn.conv2d(conv8, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv8 = tf.nn.relu6(bias, name=scope.name)
    print ("bias")
    print (bias.get_shape())

    avg_pool7 = tf.nn.avg_pool(conv8,[1,8,8,1],[1,7,7,1],'VALID',name=scope.name)
    print ("avg_pool7")
    print (avg_pool7.get_shape())

  
  # conv10
  with tf.variable_scope('softmax_linear') as scope:
    shape = avg_pool7.get_shape().as_list()
    weights = _variable_with_weight_decay('weights',shape=[10,10],stddev=0.01,wd=0.0)
    biases = _variable_on_cpu('biases', [10], tf.constant_initializer(0.0))
    reshape = tf.reshape(avg_pool7, [-1, 10])
    softmax_linear = tf.add(tf.matmul(reshape, weights),biases,name=scope.name)
  
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


def tower_loss(scope):
  """Calculate the total loss on a single tower running the CIFAR model.
  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Get images and labels for CIFAR-10.
  images, labels = read_and_preprocess_images()

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

def read_images_from_disk(img_label_Q):
      label = img_label_Q[1]
      file_contents = tf.read_file(img_label_Q[0])
      example = tf.image.decode_png(file_contents, channels=3,dtype=tf.uint8)
      label = tf.one_hot(label, 10, dtype=tf.float32)
      example.set_shape([32, 32, 3])
      label.set_shape = ([10])
      return example, label

def read_and_preprocess_images():
      data_dir="/home/ubuntu/mytf/data/X_train.txt"
      labels_dir="/home/ubuntu/mytf/data/y_train.txt"
      filenames = np.genfromtxt(data_dir,dtype="string")
      print ("read the filenames!!")

      image_list = []
      for i in xrange(len(filenames)):
        fn = filenames[i].replace('\n','').strip()
        image_list.append("/home/ubuntu/mytf/images/%s.png" % fn)

      label_list = np.genfromtxt(data_dir)
      print ("read the labellist!!")
      
      
      images = tf.convert_to_tensor(image_list, dtype=tf.string)
      labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
      print ("converted to tensors!")

      # Makes an input queue
      input_queue = tf.train.slice_input_producer([images, labels])
      print ("sliciing done")

      #Read individual image, label from the queue
      image, label = read_images_from_disk(input_queue)
      print ("read_images_from_disk done")
      print (image.get_shape())
      print (label.get_shape())
      image = tf.cast(image, tf.float32)
      label = tf.cast(label, tf.float32)

      # Randomly crop a [height, width] section of the image.
      #distorted_image = tf.random_crop(image, [24, 24, 3])

      # Randomly flip the image horizontally.
      distorted_image = tf.image.random_flip_left_right(image)

      # Because these operations are not commutative, consider randomizing
      # the order their operation.
      distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
      distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)

      # Subtract off the mean and divide by the variance of the pixels.
      float_image = tf.image.per_image_whitening(distorted_image)


      min_fraction_of_examples_in_queue = 0.4
      min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
      print ('Filling queue with %d CIFAR images before starting to train. This will take a few minutes.' % min_queue_examples)

      # Generate a batch of images and labels by building up a queue of examples.
      num_preprocess_threads=8
      images, label_batch = tf.train.batch([float_image,label], batch_size=BATCH_SIZE, capacity=min_queue_examples+ 3 * BATCH_SIZE,num_threads=num_preprocess_threads)
      print ("queueuing done")
      return images, label_batch 


def train():
    #images_labels = []
    
    #images_labels = np.column_stack((y_train, X_train))
      
    
    with tf.Graph().as_default(), tf.device('/cpu:0'):
      #images = tf.constant(X_train,dtype='float32')
      #labels = tf.constant(y_train,dtype='float32')
      #images_labels = tf.constant(images_labels,dtype='float32')
      
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
      #opt = tf.train.AdamOptimizer(lr)

      #Calculate the gradients for each model tower.
      tower_grads = []
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
            print('Working on: %s_%d' % (TOWER_NAME, i))
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            loss = tower_loss(scope)

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


      # Start running operations on the Graph. allow_soft_placement must be set to
      # True to build towers on GPU, as some of the ops do not have GPU
      # implementations.
      config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=FLAGS.log_device_placement)
      config.gpu_options.allocator_type = 'BFC'
      sess = tf.Session(config=config)
      sess.run(init)

      # Start the queue runners.
      tf.train.start_queue_runners(sess=sess)

      for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 50 == 0:
          num_examples_per_step = BATCH_SIZE * FLAGS.num_gpus
          examples_per_sec = num_examples_per_step / float(duration)
          sec_per_batch = float(duration) / FLAGS.num_gpus

          format_str = ('%s: step %d, loss = %.7f (%.1f examples/sec; %.3f ''sec/batch)')
          print (format_str % (datetime.now(), step, loss_value,examples_per_sec, sec_per_batch))

        
        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
  train()


if __name__ == '__main__':
  main()

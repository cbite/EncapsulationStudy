# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 19:48:44 2019

@author: 20190294
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import numpy
import PIL, codecs, json, array, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import gzip
import os
import sys
import urllib
import tensorflow.python.platform
import numpy


FLAGS = None

WORK_DIRECTORY = 'data'
IMAGE_SIZE = 56
NUM_CHANNELS = 1
PIXEL_DEPTH = 1
alp_percentile=np.array([50])
NUM_LABELS = len(alp_percentile)+1
 # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 10
NUM_EPOCHS = 10
VALIDATION_SIZE= 20
NUM_CHANNELS = 1


##Some parameters
#TrainSize=10000




#ALPSEP=np.array([300])

#LEARNING_RATE = 0.001
#TRAIN_STEPS = 2500


##define parameters 

# Load features numbers to mutetate
FeaturesData = pd.read_csv('/media/alex/DATA/projects/Genetic Algorithms/Analysis/All_FeaturesAlp_stat.csv', 
                             delimiter='\t')
##select only low variable data
#FeaturesData_sel=FeaturesData[FeaturesData.RelTrSD<0.93].sample(frac=1)
#FeaturesData_sel=FeaturesData[((FeaturesData['ALPTrMean']<259) | (FeaturesData['ALPTrMean']>400))&(FeaturesData.RelTrSD<0.6)].sample(frac=1)

FeaturesData_sel=FeaturesData[((FeaturesData['ALPTrMean']<215) | (FeaturesData['ALPTrMean']>455.44))&(FeaturesData.RelTrSD<0.6)].sample(frac=1)


#FeaturesData_sel=FeaturesData[FeaturesData.RelTrSD<0.573].sample(frac=1)


plt.hist(FeaturesData_sel.ALPTrMean)

#TrainingSize=round(len(FeaturesData_sel)*0.7,0)
#TestingSize=len(FeaturesData_sel)-TrainingSize

TrainingSize=120
TestingSize=30

                
#ValidationSize=len(FeaturesData_sel)-(TrainingSize+TestingSize+1)

AlpTraining=np.zeros((TrainingSize,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)).astype(numpy.float32)
AlpTesting=np.zeros((TestingSize,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS)).astype(numpy.float32)
#AlpValidation=np.zeros((ValidationSize,ImageSize))

AlpTrainingL=np.zeros((TrainingSize,1)).astype(numpy.float32)
AlpTestingL=np.zeros((TestingSize,1)).astype(numpy.float32)
#AlpValidationL=np.zeros((ValidationSize,1)).astype(numpy.float32)

k=0
l=0
m=0
ii=0
##replace orifinal data with one hot vector.

AlpData_temp=FeaturesData_sel[["FeatureIdx","ALPTrMean"]]
##binn the data
ALPSEP=numpy.percentile(AlpData_temp.ix[:,1],alp_percentile)

#xx=np.digitize(AlpData_temp.ix[:,1], numpy.percentile(AlpData_temp.ix[:,1],alp_percentile))
xx=np.digitize(AlpData_temp.ix[:,1], ALPSEP)
plt.hist(xx)
np.unique(xx)
AlpData_temp.ix[:,1]=xx

#AlpData_temp.ix[:,1]=(AlpData_temp.ix[:,1]*10/max(AlpData_temp.ix[:,1])).round()

#np.unique(AlpData_temp.ix[:,1])

#import tensorflow as tf
#idx_0 = tf.placeholder(tf.int64, [None])
#mask = tf.one_hot(idx_0, depth=10, on_value=1, off_value=0, axis=-1)
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#a = sess.run([mask],feed_dict={idx_0:[3]})
#print(a)
#
#xxx=tf.one_hot(2, depth=10, on_value=1, off_value=0, axis=-1, dtype=None, name=None)
#dataL=AlpTestingL
#
#xxx=(numpy.arange(NUM_LABELS) == dataL).astype(numpy.float32)
##convert labels to One hot

#tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
#FLAGS = tf.app.flags.FLAGS


def extract_1hot(dataL):
  # Convert to dense 1-hot representation.
  #return (numpy.arange(NUM_LABELS) == dataL[:, None]).astype(numpy.float32)
  return (numpy.arange(NUM_LABELS) == dataL).astype(numpy.float32)

## Show results
def display_surface(num):
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([56,56])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_mult_flat(start, stop):
    images = x_train[start].reshape([1,IMAGE_SIZE])
    for i in range(start+1,stop):
        images = np.concatenate((images, x_train[i].reshape([1,IMAGE_SIZE])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()

def next_batch(num, data1,data2):
    """
    Return a total of `num` samples from the array `data`. 
    """
    idx = np.arange(0, len(data1))  # get all possible indexes
    np.random.shuffle(idx)  # shuffle indexes
    idx = idx[0:num]  # use only `num` random indexes
    data_shuffle1 = [data1[i] for i in idx]  # get list of `num` random samples
    data_shuffle1 = np.asarray(data_shuffle1)  # get back numpy array
    data_shuffle2 = [data2[i] for i in idx]  # get list of `num` random samples
    data_shuffle2 = np.asarray(data_shuffle2)  # get back numpy array
    return data_shuffle1, data_shuffle2

## rotate and reverse surfaces to increase representation
#iiz=0


def surf_mult (data):
    ##rotate
    surf_rot=data
    for surf in data:
        #surf=FeatImgGene.reshape(IMAGE_SIZE, IMAGE_SIZE)
        surfr90=np.rot90(surf, k=1)
        surfr180=np.rot90(surf, k=2)
        surfr270=np.rot90(surf, k=3)
        surf_rot=np.vstack((surf_rot,np.stack((surfr90,surfr180, surfr270))))
    ##mirror everything
    surf_rot_flip=np.empty(shape=(1, IMAGE_SIZE, IMAGE_SIZE, 1))
    for surfr in surf_rot:
        surff=np.fliplr(surfr)
        surf_rot_flip=np.vstack((surf_rot_flip, np.stack((surfr,surff))))
            
    return(np.delete(surf_rot_flip,0,0).astype(numpy.float32))

def label_mult (dataL):
   ##rotate
    surf_rot=dataL
    for surf in dataL:
        #surf=FeatImgGene.reshape(IMAGE_SIZE, IMAGE_SIZE)
        surfr90=surf
        surfr180=surf
        surfr270=surf
        surf_rot=np.vstack((surf_rot,np.stack((surfr90,surfr180, surfr270))))
    ##mirror everything
    surf_rot_flip=np.empty(shape=(1, NUM_LABELS))
    for surfr in surf_rot:
        surff=surfr
        surf_rot_flip=np.vstack((surf_rot_flip, np.stack((surfr,surff))))
            
    return(np.delete(surf_rot_flip,0,0).astype(numpy.float32))
    
        



#def extract_data(filename, num_images):
#  """Extract the images into a 4D tensor [image index, y, x, channels].
#  Values are rescaled from [0, 255] down to [-0.5, 0.5].
#  """
#  print('Extracting', filename)
#  with gzip.open(filename) as bytestream:
#    bytestream.read(16)
#    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
#    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
#    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
#    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
#    return data
#
#
#def extract_labels(filename, num_images):
#  """Extract the labels into a 1-hot matrix [image index, label index]."""
#  print('Extracting', filename)
#  with gzip.open(filename) as bytestream:
#    bytestream.read(8)
#    buf = bytestream.read(1 * num_images)
#    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
#  # Convert to dense 1-hot representation.
#  return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)
#
#
#def fake_data(num_images):
#  """Generate a fake dataset that matches the dimensions of MNIST."""
#  data = numpy.ndarray(
#      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
#      dtype=numpy.float32)
#  labels = numpy.zeros(shape=(num_images, NUM_LABELS), dtype=numpy.float32)
#  for image in xrange(num_images):
#    label = image % 2
#    data[image, :, :, 0] = label - 0.5
#    labels[image, label] = 1.0
#  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and 1-hot labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
      predictions.shape[0])



##Load all images as  genes
for i in FeaturesData_sel.FeatureIdx.values:
    FeatImg = np.asarray(PIL.Image.open('/media/alex/DATA/projects/Genetic Algorithms/Analysis/for_deep_learning/Pattern_FeatureIdx_{}.bmp'.format(i)).convert("L"))
    FeatImg.setflags(write=1)
    FeatImg[FeatImg>0]=1
    FeatImgGene=FeatImg.ravel().astype(numpy.float32)
    ##invert to have space between patterns as input
    FeatImgGene=(np.invert(FeatImgGene.astype(bool))).astype(numpy.float32)
    
    FeatImgGene= FeatImgGene.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
    if ii<TrainingSize:
        AlpTraining[k,:]=FeatImgGene
        AlpTrainingL[k,:]=AlpData_temp.ALPTrMean[AlpData_temp.FeatureIdx==i].values.astype(numpy.float32)
        k+=1
    if ii>TrainingSize:
        AlpTesting[l,:]=FeatImgGene
        AlpTestingL[l,:]=AlpData_temp.ALPTrMean[AlpData_temp.FeatureIdx==i].values.astype(numpy.float32)
        l+=1
    ii+=1

##convert labels to One hot
AlpTrainingL1=extract_1hot(AlpTrainingL).astype(numpy.float32).astype(numpy.float32)
AlpTestingL1=extract_1hot(AlpTestingL).astype(numpy.float32).astype(numpy.float32)
#AlpValidationL1=extract_1hot(AlpValidationL)

FLAGS = None


##Perform analysis

def main(argv=None):  # pylint: disable=unused-argument
  #  if FLAGS.self_test:
#    print('Running self-test.')
#    train_data, train_labels = fake_data(256)
#    validation_data, validation_labels = fake_data(16)
#    test_data, test_labels = fake_data(256)
#    num_epochs = 1
  #  else:
    # Get the data.

    # Extract it into numpy arrays.
  train_data = AlpTraining
  train_labels = AlpTrainingL1
  test_data = surf_mult(AlpTesting)
  test_labels = label_mult(AlpTestingL1)
    
  # Generate a validation set.
    
  idxi = np.random.randint((len(train_data)-1),size=VALIDATION_SIZE)  # get all possible indexes
  mask = np.ones(len(train_data), dtype=bool)
  mask[[idxi]] = False
    
    
    ##Fenerate random indexes for Validation set
  validation_data = surf_mult(train_data[np.invert(mask), :, :, :])
  validation_labels = label_mult(train_labels[np.invert(mask)])
  train_data = surf_mult(train_data[mask, :, :, :])
  train_labels = label_mult(train_labels[mask])
  num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]


def train():
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir,
                                    one_hot=True,
                                    fake_data=FLAGS.fake_data)

  sess = tf.InteractiveSession()
  # Create a multilayer model.

  # Input placeholders
  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

  with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

  # We can't initialize these variables to 0 - the network will get stuck.
  def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations

  hidden1 = nn_layer(x, 784, 500, 'layer1')

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

  # Do not apply softmax activation yet, see below.
  y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

  with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the
    # raw outputs of the nn_layer above, and then average across
    # the batch.
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
      cross_entropy = tf.reduce_mean(diff)
  tf.summary.scalar('cross_entropy', cross_entropy)

  with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
        cross_entropy)

  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  tf.global_variables_initializer().run()

  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
  train_writer.close()
  test_writer.close()


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
  parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--log_dir', type=str, default='/tmp/tensorflow/mnist/logs/mnist_with_summaries',
                      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 19:50:21 2019

@author: 20190294
"""



##Some parameters
TrainSize=10000

alp_percentile=np.array([50])

ALPSEP=numpy.percentile(AlpData_temp.ix[:,1],alp_percentile)
ALPSEP=np.array([300])
NUM_LABELS=len(alp_percentile)+1
ImageSizeX=56
LEARNING_RATE = 0.001
TRAIN_STEPS = 2500
ImageSize=56*56

##define parameters 

# Load features numbers to mutetate
FeaturesData = pd.read_csv('/media/alex/DATA/projects/Genetic Algorithms/Analysis/All_FeaturesAlp_stat.csv', 
                             delimiter='\t')
##select only low variable data
#FeaturesData_sel=FeaturesData[FeaturesData.RelTrSD<0.6].sample(frac=1)
FeaturesData_sel=FeaturesData[((FeaturesData['ALPTrMean']<200) | (FeaturesData['ALPTrMean']>500))&(FeaturesData.RelTrSD<0.6)].sample(frac=1)

plt.hist(FeaturesData_sel.ALPTrMean)

TrainingSize=round(len(FeaturesData_sel)*0.7,0)
TestingSize=round(len(FeaturesData_sel)*0.2,0)
ValidationSize=len(FeaturesData_sel)-(TrainingSize+TestingSize+1)

AlpTraining=np.zeros((TrainingSize,ImageSize))
AlpTesting=np.zeros((TestingSize,ImageSize))
AlpValidation=np.zeros((ValidationSize,ImageSize))

AlpTrainingL=np.zeros((TrainingSize,1))
AlpTestingL=np.zeros((TestingSize,1))
AlpValidationL=np.zeros((ValidationSize,1))

k=0
l=0
m=0
ii=0
##replace orifinal data with one hot vector.

AlpData_temp=FeaturesData_sel[["FeatureIdx","ALPTrMean"]]
##binn the data


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
def extract_1hot(dataL):
  # Convert to dense 1-hot representation.
  #return (numpy.arange(NUM_LABELS) == dataL[:, None]).astype(numpy.float32)
  return (numpy.arange(NUM_LABELS) == dataL).astype(numpy.float32)

## Show results
def display_surface(num):
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_mult_flat(start, stop):
    images = x_train[start].reshape([1,ImageSize])
    for i in range(start+1,stop):
        images = np.concatenate((images, x_train[i].reshape([1,ImageSize])))
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





##Load all images as  genes
for i in FeaturesData_sel.FeatureIdx.values:
    FeatImg = np.asarray(PIL.Image.open('/media/alex/DATA/projects/Genetic Algorithms/Analysis/for_deep_learning/Pattern_FeatureIdx_{}.bmp'.format(i)).convert("L"))
    FeatImg.setflags(write=1)
    FeatImg[FeatImg>0]=1
    FeatImgGene=FeatImg.ravel().astype(numpy.float32)
    if ii<TrainingSize:
        AlpTraining[k,:]=FeatImgGene
        AlpTrainingL[k,:]=AlpData_temp.ALPTrMean[AlpData_temp.FeatureIdx==i].values.astype(numpy.float32)
        k+=1
    if ii>TrainingSize and ii<(TrainingSize+TestingSize+1):
        AlpTesting[l,:]=FeatImgGene
        AlpTestingL[l,:]=AlpData_temp.ALPTrMean[AlpData_temp.FeatureIdx==i].values.astype(numpy.float32)
        l+=1
    if ii>(TrainingSize+TestingSize):
        AlpValidation[m,:]=FeatImgGene
        AlpValidationL[m,:]=AlpData_temp.ALPTrMean[AlpData_temp.FeatureIdx==i].values.astype(numpy.float32)
        m+=1
    ii+=1

##convert labels to One hot
AlpTrainingL1=extract_1hot(AlpTrainingL)
AlpTestingL1=extract_1hot(AlpTestingL)
AlpValidationL1=extract_1hot(AlpValidationL)




##Perform analysis
import argparse
import sys

#from tensorflow.examples.tutorials.mnist import input_data

##make function for nextbatch
x_test=AlpTesting
y_test=AlpTestingL1


#def main(_):
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, ImageSize])
y_ = tf.placeholder(tf.float32, shape=[None, NUM_LABELS])

W = tf.Variable(tf.zeros([ImageSize,NUM_LABELS]))
b = tf.Variable(tf.zeros([NUM_LABELS]))
y = tf.nn.softmax(tf.matmul(x,W) + b)



cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
training = tf.train.AdadeltaOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(TrainSize):
    #x_train, y_train = next_batch(50,AlpTraining,AlpTrainingL1)
    x_train=AlpTraining
    y_train=AlpTrainingL1
    sess.run(training, feed_dict={x: x_train, y_: y_train})
    print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) + '  Loss = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})))
for i in range(NUM_LABELS):
    plt.subplot(2, 5, i+1)
    weight = sess.run(W)[:,i]
    plt.title(i)
    plt.imshow(weight.reshape([56,56]), cmap=plt.get_cmap('seismic'))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

plt.figure()
plt.show()

sess.close()

    
#main(_)   
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
#                        help='Directory for storing input data')
#    FLAGS, unparsed = parser.parse_known_args()
#    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
#
 
    
 
    
    

#
#
#
#
#
#      
###Define function to calculate error
#def calc(x, y):
## Returns predictions and error
#    predictions = tf.add(b, tf.matmul(x, w))
#    error = tf.reduce_mean(tf.square(y - predictions)) 
#    return [ predictions, error ]
#
#y, cost = calc(train_features, train_prices)
## Feel free to tweak these 2 values:
#learning_rate = 0.025
#epochs = 3000
#points = [[], []] # You'll see later why I need this
#
#init = tf.global_variables_initializer()
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  
#
#
#with tf.Session() as sess:
#
#    sess.run(init)
#
#    for i in list(range(epochs)):
#
#        sess.run(optimizer)
#
#        if i % 10 == 0.:
#            points[0].append(i+1)
#            points[1].append(sess.run(cost))
#
#        if i % 100 == 0:
#            print(sess.run(cost))
#
#    plt.plot(points[0], points[1], 'r--')
#    plt.axis([0, epochs, 50, 600])
#    plt.show()
#
#    valid_cost = calc(valid_features, valid_prices)[1]
#
#    print('Validation error =', sess.run(valid_cost), '\n')
#
#    test_cost = calc(test_features, test_prices)[1]
#
#    # print('Test error =', sess.run(test_cost), '\n')
#view raw
# 

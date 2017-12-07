# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import vgg
import math
import tensorflow as tf
import numpy as np
from keras.datasets import cifar10
from sys import stderr
import scipy.misc

import StringIO
import os
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
import copy
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)
pooling = 'max'
meanImage = np.mean(xTrain, axis=0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage




#CONTENT_LAYERS = ('relu4_2', 'relu5_2')
CONTENT_LAYERS = ['relu5_4']#,'relu1_2']
target_w = ['conv5_4']
prune_percent = {'conv5_4':20}
last_layer = 'relu5_4'
try:
    reduce
except NameError:
    from functools import reduce

def save_response(weights,name):
    # shape, weight_arr, name
    shape= weights.shape
    np.save(name,weights)    
    # by the way what is the shape of ..?


network = 'imagenet-vgg-verydeep-19.mat'

vgg_weights, vgg_mean_pixel = vgg.load_net(network)
vgg_weights_2 = copy.deepcopy(vgg_weights)
deviceType = "/cpu:0"
tf.reset_default_graph()
with tf.device(deviceType):
    x = tf.placeholder(tf.float32, [None,2,2,512])
    y = tf.placeholder(tf.int64,[None])   
def buildModel(vgg_weights):
    with tf.device(deviceType):
    # not sure how to do this yet    
        #x = tf.reshape(x,[-1,2*2*512])
        x2 = tf.reshape(x,[-1,2*2*512])
        w = tf.get_variable('w',shape = [2*2*512,10])
        b = tf.get_variable('b',shape = [10])
        yOut = tf.matmul(x2,w)+b
        totalLoss = tf.losses.hinge_loss(tf.one_hot(y, 10), logits=yOut)
        meanLoss = tf.reduce_mean(totalLoss)
        optimizer = tf.train.AdamOptimizer(5e-4)
        trainStep = optimizer.minimize(meanLoss)

            # Define correct Prediction and accuracy                                                   
        correctPrediction = tf.equal(tf.argmax(yOut, 1), y)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
        return [meanLoss,accuracy,trainStep]
def train(sess,image,net,Model, xT, yT, xV, yV, xTe, yTe, batchSize=5, epochs=2, printEvery=1):
    trainIndex = np.arange(xTrain.shape[0])
#    sess.run(tf.global_variables_initializer())
    np.random.shuffle(trainIndex)
    with tf.device(deviceType):
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            # Mini-batch                                                                                            
            losses = []
            accs = []
           # For each batch in training data                                                                       
            for i in range(int(math.ceil(xTrain.shape[0] / batchSize))):
                # Get the batch data for training                                                                   
                startIndex = (i * batchSize) % xTrain.shape[0]
                idX = trainIndex[startIndex:startIndex + batchSize]
                currentBatchSize = yTrain[idX].shape[0]
                # get response from those train data
                responseT = net[last_layer].eval(feed_dict={image:xT[idX,:]})


                #print type(responseT)
                #print responseT
                # Train                                                                                             
                loss, acc, _ = sess.run(Model, feed_dict={x: responseT.astype(np.float32), y: yT[idX]})

                # Collect all mini-batch loss and accuracy                                                          
                losses.append(loss * currentBatchSize)
                accs.append(acc * currentBatchSize)

            totalAcc = np.sum(accs) / float(xTrain.shape[0])
            totalLoss = np.sum(losses) / xTrain.shape[0]
            if e % printEvery == 0:
                #print('Iteration {0}: loss = {1:.3f} and training accuracy = {2:.2f}%,'.format(e, totalLoss, totalAcc * 100), end='')    
                print('Iteration {0}: loss = {1:.3f} and training accuracy = {2:.2f}%,'.format(e, totalLoss, totalAcc * 100))
                responseV = net[last_layer].eval(feed_dict={image:xV})
                loss, acc = sess.run(Model[:-1], feed_dict={x: responseV, y: yV})
                print(' Validate loss = {0:.3f} and validate accuracy = {1:.2f}%'.format(loss, acc * 100))
        responseTe = net[last_layer].eval(feed_dict={image:xTe})
        loss, acc = sess.run(Model[:-1], feed_dict={x: responseTe, y: yTe})
        print('Testing loss = {0:.3f} and testing accuracy = {1:.2f}%'.format(loss, acc * 100))
def test():
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """


    # compute content features in feedforward mode

    with  tf.Session() as sess:
        image = tf.placeholder('float', shape=[None,32,32,3])
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        
        
        #content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        content = xVal[0,:,:,:]
        content = content.reshape((1,)+ content.shape)
        print content.shape

        # retrain happens here

        train(sess,image,net,buildModel(vgg_weights),xTrain, yTrain, xVal, yVal, xTest, yTest)
        for weight_name,weight in net.items():
            if weight_name in target_w:

                filename = 'complete_%s'%(weight_name)
                content_mine = np.zeros(shape)
                content_mine[0] = content_pre
                save_response(weight.eval(feed_dict={image:content_mine}),filename)


    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights_2, image, pooling,apply_pruning=True,target_w = target_w,prune_percent = prune_percent)
        
        #content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])

        train(sess,image,net,buildModel(vgg_weights2),xTrain, yTrain, xVal, yVal, xTest, yTest)

        for weight_name,weight in net.items():
            if weight_name in target_w:
                filename = 'pruned_%s_%s'%(weight_name,str(prune_percent[weight_name]))

                content_mine = np.zeros(shape)
                content_mine[0] = content_pre
                save_response(weight.eval(feed_dict={image:content_mine}),filename)    
    print "done"
    
test()
def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

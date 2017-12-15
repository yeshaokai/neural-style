# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)


def load_net(data_path):
    data = scipy.io.loadmat(data_path)
    if not all(i in data for i in ('layers', 'classes', 'normalization')):
        raise ValueError("You're using the wrong VGG19 data. Please follow the instructions in the README to download the correct data.")
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    return weights, mean_pixel
def prune_weight(weight_arr,weight_name,prune_percent):                                                                                
  percent = prune_percent[weight_name]
  non_zero_weight_arr = weight_arr[weight_arr!=0]
  pcen = np.percentile(abs(non_zero_weight_arr),percent)
  print ("percentile " + str(pcen))
  under_threshold = abs(weight_arr)< pcen
  above_threshold = abs(weight_arr)>= pcen
  before = len(non_zero_weight_arr)
  weight_arr[under_threshold] = 0
  #weight_arr[above_threshold] = 0
  non_zero_weight_arr = weight_arr[weight_arr!=0]
  after = len(non_zero_weight_arr)
  
  return [above_threshold,weight_arr]
def apply_prune(name,kernels,target_w,prune_percent):
    if name in target_w:
        print ("at weight "+name)
        weight_arr = kernels
        print ("before pruning #non zero parameters " + str(np.sum(kernels!=0)))
        before = np.sum(weight_arr!=0)
        mask,weight_arr_pruned = prune_weight(weight_arr,name,prune_percent)
        after = np.sum(kernels!=0)
        print ("pruned "+ str(before-after))    
        print ("after prunning #non zero parameters " + str(np.sum(kernels!=0)))

    


def net_preloaded(weights, input_image, pooling,apply_pruning = False,target_w = None, prune_percent = None):
    net = {}
    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            if apply_pruning and target_w and prune_percent:
                apply_prune(name,kernels,target_w,prune_percent)
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias,name)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, pooling)
        net[name] = current


    assert len(net) == len(VGG19_LAYERS)
    return net

def _conv_layer(input, weights, bias,name):
#    print "at " + name
#    print np.sum(weights!=0)
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    if name == 'conv_5_4':
        return conv
    else:
        return  tf.nn.bias_add(conv, bias)


def _pool_layer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')

def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel

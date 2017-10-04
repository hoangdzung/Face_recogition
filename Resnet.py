from __future__ import print_function
import tensorflow as tf
import numpy as np 
from dataset import Dataset
import math
import os
import cv2 
import sys

class Resnet():
    def __init__(self, name, input_shape, n_classes, n_features = 4096, batch_size = 32, learning_rate = 0.01):
        """
            name: string, name of model using for saving graph
            input_shape: tuple, size of image
            n_classes: int, number of classes
            n_features: int, size of embedded features
            batch_size: int, number of images of a batch
            learning_rate: float, learning rate of opitmizer
        
        """
        print("resnet_init")
        self.name = name
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_features = n_features
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype = tf.float32, trainable = False, name = "global_step")
    
    def build(self):
        raise NotImplementedError
        
    def _create_placeholder(self):
        self.input_matrix = tf.placeholder(dtype = tf.float32,
                                    shape = [None,
                                            self.input_shape[0],
                                            self.input_shape[1],
                                            self.input_shape[2]],
                                    name = 'input')
        self.label_one_hot = tf.placeholder(dtype = tf.float32, shape = [None, self.n_classes])

    @staticmethod
    def new_Weight(shape, stddev = 0.01):
        return tf.Variable(tf.truncated_normal(shape, stddev = stddev))

    @staticmethod
    def new_Bias(shape):
        return tf.Variable(tf.constant(0.01, shape=[shape]))
    
    def _conv2d(self, input, filter_size, n_filters, strides, padding = 'VALID', use_relu = False):
        n_input_channels = int(input.shape[-1])
        shape = [filter_size, filter_size, n_input_channels, n_filters]
        
        weights = self.new_Weight(shape, stddev = 3.0/math.sqrt(n_input_channels + n_filters))
        biases = self.new_Bias(n_filters)

        layer = tf.nn.conv2d(input = input,
                            filter = weights,
                            strides = [1, strides, strides, 1],
                            padding = padding)
        layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("layer", layer)
        return layer
    

    def _maxpooling2d_layer(self, input, size, strides):
        layer = tf.nn.max_pool(value = input,
                            ksize=[1, size, size, 1],
                            strides=[1, strides, strides, 1],
                            padding='VALID')
        return layer
    
    def _batch_norm(self, input):
        batch_mean, batch_var = tf.nn.moments(input ,[0])
        depth = int(input.shape[-1])
        scale = self.new_Bias(depth)
        beta = self.new_Bias(depth)
        layer = tf.nn.batch_normalization(input, batch_mean, batch_var , beta, scale, 1e-3)
        return layer

    def _activation_relu(self, input):
        return tf.nn.relu(input)
    
    def _identity_block(self, input, kernel_size, filters):
        n_filters1, n_filters2, n_filters3 = filters
        
        x1 = self._conv2d(input, filter_size = 1, n_filters = n_filters1, strides = 1)
        x2 = self._batch_norm(x1)
        x3 = self._activation_relu(x2)

        x4 = self._conv2d(x3, filter_size = kernel_size, n_filters = n_filters2, strides = 1, padding = 'SAME')
        x5 = self._batch_norm(x4)
        x6 = self._activation_relu(x5)

        x7 = self._conv2d(x6, filter_size = 1, n_filters = n_filters3, strides = 1)
        x8 = self._batch_norm(x7)

        x9 = tf.add(x8, input)
        x10 = self._activation_relu(x9)
        return x10
    
    def _conv_block(self, input, kernel_size, filters, strides = 2):
        n_filters1, n_filters2, n_filters3 = filters
        
        x1 = self._conv2d(input, filter_size = 1, n_filters = n_filters1, strides = strides)
        x2 = self._batch_norm(x1)
        x3 = self._activation_relu(x2)

        x4 = self._conv2d(x3, filter_size = kernel_size, n_filters = n_filters2, strides = 1, padding = 'SAME')
        x5 = self._batch_norm(x4)
        x6 = self._activation_relu(x5)

        x7 = self._conv2d(x6, filter_size = 1, n_filters = n_filters3, strides = 1)
        x8 = self._batch_norm(x7)

        shortcut1 = self._conv2d(input, filter_size = 1, n_filters = n_filters3, strides = strides)
        shortcut2 = self._batch_norm(shortcut1)

        x9 = tf.add(x8, shortcut2)
        x10 = self._activation_relu(x9)
        return x10

    def _flatten_layer(self, input):
        h = int(input.shape[1])
        w = int(input.shape[2])
        d = int(input.shape[3])
        return tf.reshape(input, [-1, h*w*d])

    def _fully_connected_layer(self, input, n_outputs, activation = None, drop_rate = None):
        n_inputs = int(input.shape[-1])
        weights = self.new_Weight([n_inputs, n_outputs], stddev = 3.0/math.sqrt(n_inputs+n_outputs))
        biases = self.new_Bias(n_outputs)

        layer = tf.matmul(input, weights) + biases

        if activation == 'softmax':
            layer = tf.nn.softmax(layer)
        elif activation == 'relu':
            layer = tf.nn.relu(layer)
        if drop_rate != None:
            layer = tf.nn.dropout(layer, drop_rate)
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("layer", layer)
        return layer

    def _build_model(self):
        self.input_padding = tf.pad(self.input_matrix, [[0, 0], [3, 3], [3, 3], [0, 0]])
        
        self.conv1 = self._conv2d(self.input_padding, filter_size = 7, n_filters = 64, strides = 2)
        self.batch_norm1 = self._batch_norm(self.conv1)
        self.activation1 = self._activation_relu(self.batch_norm1)
        self.maxpooling1 = self._maxpooling2d_layer(self.activation1, size = 3, strides = 2)

        self.conv_block2 = self._conv_block(self.maxpooling1, kernel_size = 3, filters = [64, 64, 256], strides = 1)
        self.identity_block2a = self._identity_block(self.conv_block2, kernel_size = 3, filters = [64, 64, 256])
        self.identity_block2b = self._identity_block(self.identity_block2a, kernel_size = 3, filters = [64, 64, 256])

        self.conv_block3 = self._conv_block(self.identity_block2b, kernel_size = 3, filters = [128, 128, 512], strides = 2)
        self.identity_block3a = self._identity_block(self.conv_block3, kernel_size = 3, filters = [128, 128, 512])
        self.identity_block3b = self._identity_block(self.identity_block3a, kernel_size = 3, filters = [128, 128, 512])
        self.identity_block3c = self._identity_block(self.identity_block3b, kernel_size = 3, filters = [128, 128, 512])

        self.conv_block4 = self._conv_block(self.identity_block3c, kernel_size = 3, filters = [256, 256, 1024], strides = 2)
        self.identity_block4a = self._identity_block(self.conv_block4, kernel_size = 3, filters = [256, 256, 1024])
        self.identity_block4b = self._identity_block(self.identity_block4a, kernel_size = 3, filters = [256, 256, 1024])
        self.identity_block4c = self._identity_block(self.identity_block4b, kernel_size = 3, filters = [256, 256, 1024])
        self.identity_block4d = self._identity_block(self.identity_block4c, kernel_size = 3, filters = [256, 256, 1024])
        self.identity_block4e = self._identity_block(self.identity_block4d, kernel_size = 3, filters = [256, 256, 1024])

        self.conv_block5 = self._conv_block(self.identity_block4e, kernel_size = 3, filters = [512, 512, 2048], strides = 2)
        self.identity_block5a = self._identity_block(self.conv_block5, kernel_size = 3, filters = [512, 512, 2048])
        self.identity_block5b = self._identity_block(self.identity_block5a, kernel_size = 3, filters = [512, 512, 2048])

        self.avg_pool = tf.nn.avg_pool(self.identity_block5b, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID')

        self.flatten = self._flatten_layer(self.avg_pool)
        self.full1 = self._fully_connected_layer(self.flatten, 4096, 'relu', 0.5)
        self.full2 = self._fully_connected_layer(self.full1, 4096, 'relu', 0.5)
        self.embedding = tf.nn.l2_normalize(self.full2, 1)
        self.full3 = self._fully_connected_layer(self.full2, self.n_classes, 'softmax')
        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("graph_" + self.name)

    def _create_loss(self):
        raise NotImplementedError
        
    def _create_optimizer(self):
        raise NotImplementedError

    def _create_evaluater(self):
        raise NotImplementedError         

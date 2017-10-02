"""
   DeepID2: Two seperate models with shared weights
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np 
from dataset import Dataset
from fake_dataset import Fake_data
import math
import os
import cv2 
import sys
import time

class Resnet():
    def __init__(self, name, n_classes, n_features, input_shape, reuse = None):
        """
            name: string, name of model using for saving graph
            input_shape: tuple, size of image
            n_classes: int, number of classes
            n_features: int, size of embedded features

        """
        self.name = name
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_features = n_features
        self.reuse = reuse
    
    def build(self):
        self._create_placeholder()
        self._build_model()
        
    def _create_placeholder(self):
        self.input_matrix = tf.placeholder(dtype = tf.float32,
                                    shape = [None,
                                            self.input_shape[0],
                                            self.input_shape[1],
                                            self.input_shape[2]],
                                    name = 'input_matrix')
        self.label_one_hot = tf.placeholder(dtype = tf.float32, shape = [None, self.n_classes], name = 'label_one_hot')

    def _conv2d(self, input, filter_size, n_filters, strides, block, padding = 'VALID', use_relu = False):
        with tf.variable_scope("conv_"+block, reuse=self.reuse):
            n_input_channels = int(input.shape[-1])
            shape = [filter_size, filter_size, n_input_channels, n_filters]
        
            weight = tf.get_variable(name = 'weight', 
                                    dtype = tf.float32, 
                                    shape = shape,
                                    initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005))
            
            bias = tf.get_variable(name = 'bias',
                                dtype=tf.float32,
                                shape = n_filters,
                                initializer=tf.constant_initializer(0.0))
            
        layer = tf.nn.conv2d(input = input,
                            filter = weight,
                            strides = [1, strides, strides, 1],
                            padding = padding,
                            name = "conv_layer")
        layer += bias
        if use_relu:
            layer = tf.nn.relu(layer, "relu")
        return layer
    
    def _local_conv2d(self, input, filter_size, n_filters, strides, block, padding = 'VALID'):
        with tf.variable_scope("conv_local_"+block, reuse=self.reuse):
            image_patches = tf.extract_image_patches(input, [1, filter_size, filter_size, 1], [1, strides, strides, 1], [1, 1, 1, 1], padding = padding)
            image_patches_ = tf.expand_dims(image_patches, 4)

            w_shape = [int(image_patches.shape[1]), int(image_patches.shape[2]), int(image_patches.shape[3]), n_filters]
            b_shape = [int(image_patches.shape[1]), int(image_patches.shape[2]), n_filters]
            weight = tf.get_variable(name = 'weight',
                                    shape = w_shape,
                                    dtype= tf.float32,
                                    initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005))
            weight_ = tf.expand_dims(weight, 0)

            bias = tf.get_variable(name = 'bias',
                                shape = b_shape,
                                dtype = tf.float32,
                                initializer=tf.constant_initializer(0.0))

        layer = tf.multiply(weight_, image_patches_)
        layer = tf.reduce_sum(layer, axis=3) + bias

        return layer

    def _maxpooling2d_layer(self, input, size, strides, block):
#         with tf.variable_scope("maxpool_"+block, reuse=self.reuse):
        layer = tf.nn.max_pool(value = input,
                            ksize=[1, size, size, 1],
                            strides=[1, strides, strides, 1],
                            padding='VALID',
                            name = "max_pool_layer")
        return layer

    def _flatten_layer(self, input):
#         with tf.variable_scope("flatten", reuse=self.reuse):
        h = int(input.shape[1])
        w = int(input.shape[2])
        d = int(input.shape[3])
        return tf.reshape(input, [-1, h*w*d], name = 'flatten')

    def _fully_connected_layer(self, input, n_outputs, block, activation = None, drop_rate = None):
        with tf.variable_scope("fc_"+block, reuse=self.reuse):
            n_inputs = int(input.shape[-1])
            weight = tf.get_variable(name = 'weight', 
                                    dtype = tf.float32, 
                                    shape = [n_inputs, n_outputs],
                                    initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005))
            
            bias = tf.get_variable(name = 'bias',
                                dtype=tf.float32,
                                shape = n_outputs,
                                initializer=tf.constant_initializer(0.0))

        layer = tf.matmul(input, weight) + bias

        if activation == 'softmax':
            layer = tf.nn.softmax(layer, name = 'softmax')
        elif activation == 'relu':
            layer = tf.nn.relu(layer, name = 'relu')
        if drop_rate != None:
            layer = tf.nn.dropout(layer, drop_rate, name ='dropout')
        return weight, bias, layer

    def _build_model(self):
#         with tf.variable_scope("model", reuse=self.reuse):
        self.conv1 = self._conv2d(self.input_matrix, filter_size = 4, n_filters = 20, strides = 1, block ='1')
        self.maxpooling1 = self._maxpooling2d_layer(self.conv1, size = 2, strides = 2, block ='1')
        
        self.conv2 = self._conv2d(self.maxpooling1, filter_size = 3, n_filters = 40, strides = 1, block ='2')
        self.maxpooling2 = self._maxpooling2d_layer(self.conv2, size = 2, strides = 2, block ='2')
        
        self.conv3 = self._local_conv2d(self.maxpooling2, filter_size = 3, n_filters = 60, strides = 1, block ='3')
        self.maxpooling3 = self._maxpooling2d_layer(self.conv3, size = 2, strides = 2, block ='3')
        
        self.conv4 = self._local_conv2d(self.maxpooling3, filter_size = 2, n_filters = 80, strides = 1, block ='4' )

       # [.... locally_connected...]
        self.flatten = self._flatten_layer(self.conv4)      
        self.weight, self.bias, self.softmax = self._fully_connected_layer(self.flatten, self.n_classes, '1', 'softmax')
        self.check = tf.reduce_sum(self.softmax)

class Model():
    def __init__(self, name, n_classes, alpha = 0.0, n_features = 160, input_shape = (55, 47, 3), learning_rate = 0.001, momentum = 0.9):
        """
            name: string, name of model using for saving graph
            input_shape: tuple, size of image
            n_classes: int, number of classes
            n_features: int, size of embedded features
            learning_rate: float, learning rate of opitmizer
        
        """
        self.name = name
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.alpha = alpha
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.global_step = tf.Variable(0, dtype = tf.float32, trainable = False, name = "global_step")
        with tf.variable_scope('margin'):
            self.margin = tf.get_variable(name = 'margin', dtype = tf.float32, shape =(), initializer=tf.constant_initializer(2.0))
        self.model1 = Resnet('model1', n_classes = n_classes, n_features = n_features, input_shape = input_shape)
        self.model1.build()
        self.model2 = Resnet('model2', n_classes = n_classes, n_features = n_features, input_shape = input_shape, reuse = True)
        self.model2.build()
    
    def build(self):
        self._create_loss()
        self._create_optimizer()
        self._create_evaluater()
    def _create_loss(self):
        with tf.variable_scope("loss"):
            self.loss_ident_1 = tf.reduce_mean(-tf.reduce_sum(self.model1.label_one_hot * tf.log(self.model1.softmax), reduction_indices=[1]), name = "loss_id_1")
            self.loss_ident_2 = tf.reduce_mean(-tf.reduce_sum(self.model2.label_one_hot * tf.log(self.model2.softmax), reduction_indices=[1]), name = "loss_id_2")

            L2_norm = tf.nn.l2_loss(self.model1.flatten - self.model2.flatten)
            if tf.equal(self.model1.label_one_hot, self.model2.label_one_hot) is True:
                self.loss_ver = L2_norm
            else:
                self.loss_ver = tf.maximum(0.0, self.margin - L2_norm)
            
            tf.identity(self.loss_ver, name= "loss_verify")

    def _create_optimizer(self):
        # self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        self.optimizer = tf.train.AdamOptimizer()
        theta_id = [self.model1.weight, self.model1.bias]
        grad_theta_id = tf.gradients(self.loss_ident_1 + self.loss_ident_2, theta_id)

        theta_ve = [self.margin]
        grad_theta_ve = tf.gradients(self.loss_ver, theta_ve)
        grad_theta_ve = [self.alpha * i for i in grad_theta_ve]

        theta_c = [Var for Var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if Var.name.startswith('conv')]
        grad_theta_c = tf.gradients(self.loss_ident_1 + self.loss_ident_2 + 2 *self.alpha *self.loss_ver, theta_c)
        
        self.grads_and_vars = list(zip(grad_theta_id, theta_id)) + list(zip(grad_theta_ve, theta_ve)) + list(zip(grad_theta_c, theta_c))

        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)


    def _create_evaluater(self):
        correct = tf.equal(tf.argmax(self.model1.softmax, 1), tf.argmax(self.model1.label_one_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))        


def train(model, data = None, n_epoch = 100):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            for x, y in data.next_batch(mode = 'train'):
                x1 = np.expand_dims(x[0],0)
                x2 = np.expand_dims(x[1],0)
                y1 = np.expand_dims(y[0],0)
                y2 = np.expand_dims(y[1],0)
                feed_dict = {model.model1.input_matrix: x1, model.model1.label_one_hot: y1, model.model2.input_matrix: x1, model.model2.label_one_hot: y2}
                l_id1, l_id2, l_ver, _ = sess.run([model.loss_ident_1, model.loss_ident_2, model.loss_ver, model.train_op], feed_dict = feed_dict)
                print(l_id1, l_id2, l_ver)
                time.sleep(0.1)
            accuracy = 0.0
            for x, y in data.next_batch(mode = 'test', batch_size = 50):
                acc = sess.run(model.accuracy, feed_dict = {model.model1.input_matrix: x, model.model1.label_one_hot: y})
                accuracy += acc
            print("acc: ", accuracy/data.num_batch_test)
            # f = open('acc_deepid.txt', "a")
            # f.write(str(epoch) +":"+ str(acc)+'\n')
            # f.close()

if __name__ == '__main__':
    n_epoch = int(sys.argv[1])
    data = Dataset(2, n_classes = 5, size = (55, 47))
    model = Model(name = 'deepid2', n_classes = 5)
    model.build()
    train(model, data, n_epoch)
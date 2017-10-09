from __future__ import print_function
import tensorflow as tf
import numpy as np 
from dataset import Dataset
import math
import os
import cv2 
import sys

class Resnet():
    def __init__(self, name, n_classes, alpha = 0.2, n_features = 160, input_shape = (55, 47, 3), batch_size = 32, learning_rate = 0.005, momentum = 0.9, reuse = None):
        """
            name: string, name of model using for saving graph
            input_shape: tuple, size of image
            n_classes: int, number of classes
            n_features: int, size of embedded features
            batch_size: int, number of images of a batch
            learning_rate: float, learning rate of opitmizer
        
        """
        self.name = name
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.alpha = alpha
        self.n_features = n_features
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.global_step = tf.Variable(0, dtype = tf.float32, trainable = False, name = "global_step")
        self.margin = tf.Variable(0.5, dtype=tf.float32)
        self.reuse = reuse

    def build(self):
        self._create_placeholder()
        self._build_model()
        self._create_loss()
        self._create_optimizer()

    def _create_placeholder(self):
        self.input_matrix = tf.placeholder(dtype = tf.float32,
                                    shape = [2,
                                            self.input_shape[0],
                                            self.input_shape[1],
                                            self.input_shape[2]],
                                    name = 'input_matrix')
        self.label_one_hot = tf.placeholder(dtype = tf.float32, shape = [2, self.n_classes], name = 'label_one_hot')

    def _conv2d(self, input, filter_size, n_filters, strides, block, padding = 'VALID', use_relu = False):
        with tf.variable_scope("conv_"+block, reuse=self.reuse):
            n_input_channels = int(input.shape[-1])
            shape = [filter_size, filter_size, n_input_channels, n_filters]
        
            weight = tf.get_variable(name = 'weight', 
                                    dtype = tf.float32, 
                                    shape = shape,
                                    initializer=tf.truncated_normal_initializer(0.0, 0.01))
            
            bias = tf.get_variable(name = 'bias',
                                dtype=tf.float32,
                                shape = n_filters,
                                initializer=tf.constant_initializer(0.01))
            
        layer = tf.nn.conv2d(input = input,
                            filter = weight,
                            strides = [1, strides, strides, 1],
                            padding = padding,
                            name = "conv_layer")
        layer += bias
        if use_relu:
            layer = tf.nn.relu(layer, "relu")
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
                                    initializer=tf.truncated_normal_initializer(0.0, 0.01))
            
            bias = tf.get_variable(name = 'bias',
                                dtype=tf.float32,
                                shape = n_outputs,
                                initializer=tf.constant_initializer(0.01))

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
        
        self.conv3 = self._conv2d(self.maxpooling2, filter_size = 3, n_filters = 60, strides = 1, block ='3')
        self.maxpooling3 = self._maxpooling2d_layer(self.conv2, size = 2, strides = 2, block ='3')
        
        self.conv4 = self._conv2d(self.maxpooling3, filter_size = 2, n_filters = 80, strides = 1, block ='4' )

       # [.... locally_connected...]
        self.flatten = self._flatten_layer(self.conv4)      
        
        self.f1, self.f2 = tf.split(self.flatten, [1,1])

        self.weight, self.bias, self.softmax = self._fully_connected_layer(self.flatten, self.n_classes, '1', 'softmax')
        
        self.softmax1, self.softmax2 = tf.split(self.softmax, [1,1])
        self.check1 = tf.reduce_sum(self.softmax1)
        self.check2 = tf.reduce_sum(self.softmax2)
        self.label_one_hot1, self.label_one_hot2 = tf.split(self.label_one_hot, [1,1])

        self.merged_summary = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("graph " + self.name)

    def _create_loss(self):
        self.loss_ident_1 = tf.reduce_mean(-tf.reduce_sum(self.label_one_hot1 * tf.log(self.softmax1), reduction_indices=[1]))
        self.loss_ident_2 = tf.reduce_mean(-tf.reduce_sum(self.label_one_hot2 * tf.log(self.softmax2), reduction_indices=[1]))

        L2_norm = tf.nn.l2_loss(self.softmax1 - self.softmax2)
        if tf.equal(self.label_one_hot1, self.label_one_hot2) is True:
            self.loss_ver = L2_norm
        else:
            self.loss_ver = tf.maximum(0.0, self.margin - L2_norm)
        
    def _create_optimizer(self):
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        # self.optimizer = tf.train.AdamOptimizer()
        
        theta_id = [self.weight, self.bias]
        grad_theta_id = tf.gradients(self.loss_ident_1 + self.loss_ident_2, theta_id)

        theta_ver = [self.margin ]
        grad_theta_ver = tf.gradients(self.loss_ver, theta_ver)
        grad_theta_ver = [self.alpha * i for i in grad_theta_ver]

        theta_c = [Var for Var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if Var.name.startswith('conv')]
        print(theta_c)
        grad_theta_c = tf.gradients(self.loss_ident_1 + self.loss_ident_2 + 2 *self.alpha *self.loss_ver, theta_c)
        
        self.grads_and_vars = list(zip(grad_theta_id, theta_id)) + list(zip(grad_theta_ver, theta_ver)) + list(zip(grad_theta_c, theta_c))

        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars)

    def _create_evaluater(self):
        raise NotImplementedError  

def train(model, data = None, n_epoch = 100):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epoch):
            x1 = np.ones((55,47, 3))
            x2 = np.zeros((55,47, 3))
            x = np.stack((x1, x2))
            label = np.random.randint(10, size = 2)
            y = np.zeros(shape = (2,2))
            y[0][1] = 1
            y[1][0] = 1
            feed_dict = {model.input_matrix: x, model.label_one_hot: y}
            l_id1, l_id2, l_ver, sum_s1, sum_s2, _ = sess.run([model.loss_ident_1, model.loss_ident_2, model.loss_ver, model.check1, model.check2, model.train_op], feed_dict = feed_dict)
            print(l_id1, l_id2, l_ver, sum_s1, sum_s2)

if __name__ == '__main__':
    # data = Dataset()
    model = Resnet(name = 'deepid2', n_classes = 2)
    model.build()
    print(model.grads_and_vars)
    train(model)

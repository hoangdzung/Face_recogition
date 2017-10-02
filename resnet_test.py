from __future__ import print_function
import tensorflow as tf
import numpy as np 
import math
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data


class Resnet():
    def __init__(self, input_shape, n_classes, learning_rate = 1e-3):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype = tf.float32, trainable = False, name = "global_step")
        self._create_placeholder()
        self._build_model()
        self._create_loss()
        self._create_optimizer()
        self._create_evaluater()
        
    def _create_placeholder(self):
        self.input = tf.placeholder(dtype = tf.float32,
                                    shape = [None, 784],
                                    name = 'input')
        self.input_ = tf.reshape(self.input, [-1, 28, 28, 1])
        self.output = tf.placeholder(dtype = tf.float32, shape = [None, self.n_classes])

    @staticmethod
    def new_Weight(shape):
        size = 1.0
        for x in shape:
            size *= x  
        return tf.Variable(tf.truncated_normal(shape, stddev=1.0 / math.sqrt(size)))

    @staticmethod
    def new_Bias(shape):
        return tf.Variable(tf.constant(0.01, shape=[shape]))
    
    @staticmethod
    def _leaky_relu(input, alpha = 0.3):
        return tf.maximum(input, alpha *input)

    def _conv2d(self, input, filter_size, n_filters, strides, padding = 'VALID', use_relu = False):
        n_input_channels = int(input.shape[-1])
        shape = [filter_size, filter_size, n_input_channels, n_filters]
        
        weights = self.new_Weight(shape)
        biases = self.new_Bias(n_filters)

        layer = tf.nn.conv2d(input = input,
                            filter = weights,
                            strides = [1, strides, strides, 1],
                            padding = padding)
        layer += biases
        if use_relu:
            # layer = tf.nn.relu(layer)
            layer = self._leaky_relu(layer)
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
        # return tf.nn.relu(input)
        return self._leaky_relu(input)
    
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
        weights = self.new_Weight([n_inputs, n_outputs])
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
        # with tf.device('/gpu:0'):
        self.input_padding = tf.pad(self.input_, [[0, 0], [3, 3], [3, 3], [0, 0]])
        
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

        # self.conv_block4 = self._conv_block(self.identity_block3c, kernel_size = 3, filters = [256, 256, 1024], strides = 2)
        # self.identity_block4a = self._identity_block(self.conv_block4, kernel_size = 3, filters = [256, 256, 1024])
        # self.identity_block4b = self._identity_block(self.identity_block4a, kernel_size = 3, filters = [256, 256, 1024])
        # self.identity_block4c = self._identity_block(self.identity_block4b, kernel_size = 3, filters = [256, 256, 1024])
        # self.identity_block4d = self._identity_block(self.identity_block4c, kernel_size = 3, filters = [256, 256, 1024])
        # self.identity_block4e = self._identity_block(self.identity_block4d, kernel_size = 3, filters = [256, 256, 1024])

        # self.conv_block5 = self._conv_block(self.identity_block4e, kernel_size = 3, filters = [512, 512, 2048], strides = 2)
        # self.identity_block5a = self._identity_block(self.conv_block5, kernel_size = 3, filters = [512, 512, 2048])
        # self.identity_block5b = self._identity_block(self.identity_block5a, kernel_size = 3, filters = [512, 512, 2048])

        # # self.avg_pool = tf.nn.avg_pool(self.identity_block5b, ksize = [1, 7, 7, 1], strides = [1, 2, 2, 1], padding = 'VALID')

        self.flatten = self._flatten_layer(self.identity_block3c)
        self.full1 = self._fully_connected_layer(self.flatten, 4096, 'relu', 0.5)
        self.full2 = self._fully_connected_layer(self.full1, 4096, 'relu', 0.5)
        self.full3 = self._fully_connected_layer(self.full2, self.n_classes, 'softmax')
        # self.merged_summary = tf.summary.merge_all()
        # self.writer = tf.summary.FileWriter("graph")

    def _create_loss(self):
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.full3, labels = self.output))
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.output * tf.log(self.full3), reduction_indices=[1]))
        
    def _create_optimizer(self):
        # with tf.device('/gpu:0'):
        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate).minimize(self.loss, global_step = self.global_step)

    def _create_evaluater(self):
        correct = tf.equal(tf.argmax(self.full3, 1), tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

def train_model(model, n_epoch, data):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # model.writer.add_graph(sess.graph)
        for epoch in range(n_epoch):
            batch = 0
            total_loss = 0.0
            accuracy = 0.0
            n_batch_train = len(data.train.labels)/32 + 1
            while (batch < n_batch_train):
                x_train, y_train = data.train.next_batch(32)
                print(batch)
                feed_dict = {model.input: x_train, model.output: y_train}
                loss, acc, _  = sess.run([model.loss, model.accuracy, model.optimizer], feed_dict = feed_dict)
                accuracy += acc
                total_loss += loss
                batch += 1
            print("Epoch {}: loss {}, train_acc:{}".format(epoch, total_loss, accuracy/batch))
            batch = 0
            n_batch_test = len(data.test.labels)/32 + 1
            while (batch < n_batch_test):
                x_test, y_test = data.test.next_batch(32)
                feed_dict = {model.input: x_test, model.output: y_test}
                acc  = sess.run(model.accuracy, feed_dict = feed_dict)
                accuracy += acc
                batch += 1
            print("test_acc:{}".format(accuracy/batch))
            
def main():
    size = int(sys.argv[1])
    n_epoch = int(sys.argv[2])
    data = input_data.read_data_sets('data/MNIST/', one_hot=True)
    model = Resnet(input_shape = (size, size, 1), n_classes = 10)
    train_model(model, n_epoch, data)

if __name__ == '__main__':
    main()

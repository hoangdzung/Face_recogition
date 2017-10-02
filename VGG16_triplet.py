from __future__ import print_function
import tensorflow as tf
import numpy as np 
from dataset import Dataset
import os
import cv2 
import sys

class VGG16():
    def __init__(self, input_shape, n_classes, batch_size, margin = 0.5, learning_rate = 1e-4):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.margin = margin
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype = tf.float32, trainable = False, name = "global_step")
        self._create_placeholder()
        self._build_model()
        self._create_loss()
        self._create_optimizer()
        self._create_evaluater()
        
    def _create_placeholder(self):
        self.input = tf.placeholder(dtype = tf.float32,
                                    shape = [self.batch_size,
                                            self.input_shape[0],
                                            self.input_shape[1],
                                            self.input_shape[2]],
                                    name = 'input')
        self.output = tf.placeholder(dtype = tf.float32, shape = [self.batch_size, self.n_classes])

    @staticmethod
    def new_Weight(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.001))

    @staticmethod
    def new_Bias(shape):
        return tf.Variable(tf.constant(0.01, shape=[shape]))

    def _conv2d(self, input, filter_size, n_filters, strides):
        n_input_channels = int(input.shape[-1])
        shape = [filter_size, filter_size, n_input_channels, n_filters]
        
        weights = self.new_Weight(shape)
        biases = self.new_Bias(n_filters)

        layer = tf.nn.conv2d(input = input,
                            filter = weights,
                            strides = [1, strides, strides, 1],
                            padding = 'SAME')
        layer += biases
        layer = tf.nn.relu(layer)

        return layer
    

    def _maxpooling2d_layer(self, input, size, strides):
        layer = tf.nn.max_pool(value = input,
                            ksize=[1, size, size, 1],
                            strides=[1, strides, strides, 1],
                            padding='VALID')
        return layer
    
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
        return layer
    
    def cvt_to_dist_matrix(self, embedding):
        qexpand = tf.expand_dims(embedding, 1)                                                                                                         
        qTexpand = tf.expand_dims(embedding, 0)                                                                                                           
        qtile = tf.tile(qexpand,[1, self.batch_size, 1])                                                                                                                   
        qTtile = tf.tile(qTexpand, [self.batch_size, 1, 1])                                                                                                                 
        deltaQ = qtile - qTtile                                                                                                                            
        deltaQ2 = deltaQ * deltaQ                                                                                                                            
        d2Q = tf.reduce_sum(deltaQ2, 2)  
        d2Q_ = tf.sqrt(d2Q)
        return d2Q_

    def _build_model(self):
        with tf.device('/gpu:0'):
            self.conv1a = self._conv2d(self.input, filter_size = 3, n_filters = 64, strides = 2)
            self.conv1b = self._conv2d(self.conv1a, filter_size = 3, n_filters = 64, strides = 2)
            self.conv1c = self._maxpooling2d_layer(self.conv1b, size = 2, strides = 2)
            
            self.conv2a = self._conv2d(self.conv1c, filter_size = 3, n_filters = 128, strides = 1)
            self.conv2b = self._conv2d(self.conv2a, filter_size = 3, n_filters = 128, strides = 1)
            self.conv2c = self._maxpooling2d_layer(self.conv2b, size = 2, strides = 2)

            self.conv3a = self._conv2d(self.conv2c, filter_size = 3, n_filters = 256, strides = 1)
            self.conv3b = self._conv2d(self.conv3a, filter_size = 3, n_filters = 256, strides = 1)
            self.conv3c = self._conv2d(self.conv3b, filter_size = 3, n_filters = 256, strides = 1)
            self.conv3d = self._maxpooling2d_layer(self.conv3c, size = 2, strides = 2)

            self.conv4a = self._conv2d(self.conv3c, filter_size = 3, n_filters = 512, strides = 1)
            self.conv4b = self._conv2d(self.conv4a, filter_size = 3, n_filters = 512, strides = 1)
            self.conv4c = self._conv2d(self.conv4b, filter_size = 3, n_filters = 512, strides = 1)
            self.conv4d = self._maxpooling2d_layer(self.conv4c, size = 2, strides = 2)

            self.conv5a = self._conv2d(self.conv4c, filter_size = 3, n_filters = 512, strides = 1)
            self.conv5b = self._conv2d(self.conv5a, filter_size = 3, n_filters = 512, strides = 1)
            self.conv5c = self._conv2d(self.conv5b, filter_size = 3, n_filters = 512, strides = 1)
            self.conv5d = self._maxpooling2d_layer(self.conv5c, size = 2, strides = 2)

            self.flatten = self._flatten_layer(self.conv5d)
            self.full1 = self._fully_connected_layer(self.flatten, 4096, 'relu', 0.3)
            self.full2 = self._fully_connected_layer(self.full1, 512, 'relu', 0.3)
            self.full3 = self._fully_connected_layer(self.full2, self.n_classes, 'softmax')
            
        
    def _create_loss(self):
        # with tf.device('/gpu:0'):

        #     # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.full3, labels = self.output))
        #     self.loss = tf.reduce_mean(-tf.reduce_sum(self.output * tf.log(self.full3), reduction_indices=[1]))
        dist_matrix = self.cvt_to_dist_matrix(self.full2)
        self.label = tf.argmax(self.label_one_hot, axis = 1)
        triplet = []
        for i in self.num_batch:
            pos_mask = tf.equal(self.label[i], self.label)
            neg_mask = tf.logical_not(pos_mask)
            possitive = tf.boolean_mask(dist_matrix[i], pos_mask)
            negative = tf.boolean_mask(dist_matrix[i], neg_mask)
            single_loss = self.margin + possitive - negative
            triplet.append(single_loss)
        self.loss = tf.add_n(triplet)

    def _create_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate).minimize(self.loss, global_step = self.global_step)

    def _create_evaluater(self):
        correct = tf.equal(tf.argmax(self.full3, 1), tf.argmax(self.output, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

def train_model(model, n_epoch, data):
    saver = tf.train.Saver()
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint("checkpoints")
        print(ckpt)
        if ckpt:
            saver.restore(sess, ckpt)
        # writer = tf.summary.FileWriter('graph', sess.graph)
        for epoch in range(n_epoch):
            batch = 0
            total_loss = 0.0
            accuracy = 0.0
            for x_train, y_train in data.next_batch():
                feed_dict = {model.input: x_train, model.output: y_train}
                # loss, _, summary  = sess.run([model.loss, model.optimizer, model.summary_op],
                #                             feed_dict = feed_dict)
                loss, acc, _  = sess.run([model.loss, model.accuracy, model.optimizer], feed_dict = feed_dict)
                accuracy += acc
                total_loss += loss
                print('{}/{}:{}'.format(batch, data.num_batch, acc), end="\r")
                batch += 1
            print("Epoch {}: loss {}, acc:{}".format(epoch, total_loss, accuracy/data.num_batch))
            # writer.add_summary(summary, global_step = epoch)
            saver.save(sess, 'checkpoints/check_points', global_step = model.global_step, write_meta_graph=False)
            x_test, y_test = data.test_data()
            acc = sess.run(model.accuracy, feed_dict = {model.input: x_test, model.output: y_test})
            print("test_acc: {}".format(acc))

def main():
    size = int(sys.argv[1])
    data = Dataset(batch_size = 32, folder = '/mnt/3TB-disk3/data/image-processing/face2/aligned_images_DB', size = (size, size))
    model = VGG16(input_shape = (size, size, 3), n_classes = len(data.labels), batch_size = data.batch_size)
    train_model(model, 50, data)

if __name__ == '__main__':
    main()

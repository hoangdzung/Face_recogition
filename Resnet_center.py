from __future__ import print_function
import tensorflow as tf
import numpy as np 
from dataset import Dataset
import math
import os
import cv2 
import sys
from Resnet import Resnet

class Resnet_center(Resnet):
    def __init__(self, name, input_shape, n_classes, lambda_ = 0.001, momentum = 0.9, n_features = 4096, batch_size = 32, learning_rate = 0.01):
        """
            name: string, name of model using for saving graph
            input_shape: tuple, shape of image
            n_classes: int, number of classes
            momentum: float, use for momentum optimizer
            n_features: int, size of embedded features
            batch_size: int, number of images of a batch
            learning_rate: float, learning rate of opitmizer
            lambda_ : float, weight of center loss
        """
        print("start")
        Resnet.__init__(self, name = name, 
                    input_shape = input_shape, 
                    n_classes = n_classes, 
                    n_features = n_features, 
                    batch_size = batch_size, 
                    learning_rate = learning_rate)
        self.momentum = momentum
        self.lambda_ = lambda_

    def __init__(self, input_shape, n_classes, n_features = 4096, lambda_ = 0.1, learning_rate = 0.01):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_features = n_features
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, dtype = tf.float32, trainable = False, name = "global_step")   

    def _create_loss(self):
        self.centers = tf.get_variable(name = 'center', shape = [self.n_classes, self.n_features],
                                    initializer = tf.constant_initializer(0.0), trainable = False)
        
        self.label = tf.argmax(self.label_one_hot, axis = 1)
        
        self.center_matrix = tf.gather(self.centers, self.label)
        self.diff = self.center_matrix - self.full2
        self.center_loss = tf.nn.l2_loss(self.diff)
        
        self.softmax_loss = tf.reduce_mean(-tf.reduce_sum(self.label_one_hot * tf.log(self.full3), reduction_indices=[1]))
    
        # self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name ]) * 0.001
        self.loss = self.softmax_loss + self.lambda_ * self.center_loss
    
        tf.summary.scalar("center_loss", self.center_loss)
        tf.summary.scalar("softmax_loss", self.softmax_loss)
        tf.summary.scalar("loss", self.loss)
    
    def _create_update_center_op(self):
        unique_label, unique_idx, unique_count = tf.unique_with_counts(self.label)
        self.appear_times = tf.gather(unique_count, unique_idx) + 1
        self.appear_times = tf.cast(self.appear_times, tf.float32)
        self.appear_times = tf.reshape(self.appear_times, [-1, 1])
        self.update = tf.div(self.diff, self.appear_times)
        
        self.centers_update_op = tf.scatter_sub(self.centers, self.label, self.update)

    def _create_optimizer(self):
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss, global_step = self.global_step)

    def _create_evaluater(self):
        correct = tf.equal(tf.argmax(self.full3, 1), tf.argmax(self.label_one_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        tf.summary.scalar('accuracy', self.accuracy)

    def build(self):
        self._create_placeholder()
        self._build_model()
        self._create_loss()
        self._create_update_center_op()
        self._create_optimizer()
        self._create_evaluater() 

def train_model(model, n_epoch, data):
    # file_ckp = "checkpoints_" + model.name
    # saver = tf.train.Saver()
    # if not os.path.exists(file_ckp):
    #     os.mkdir(file_ckp)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # ckpt = tf.train.latest_checkpoint(file_ckp)
        # print(ckpt)
        # if ckpt:
        #     saver.restore(sess, ckpt)
        model.writer.add_graph(sess.graph)
        step = 1
        for epoch in range(n_epoch):
            batch = 0
            # total_loss = 0.0
            total_c_loss = 0.0
            total_s_loss = 0.0
            accuracy = 0.0
            for x_train, y_train in data.next_batch(mode = 'train'):
                feed_dict = {model.input_matrix: x_train, model.label_one_hot: y_train}
                loss, c_loss, s_loss, acc, Sum, _, _  = sess.run([model.loss,
                                                                model.center_loss, 
                                                                model.softmax_loss, 
                                                                model.accuracy, 
                                                                model.merged_summary, 
                                                                model.optimizer, 
                                                                model.centers_update_op], 
                                                                feed_dict = feed_dict)
                accuracy += acc
                # total_loss += loss
                total_c_loss += c_loss
                total_s_loss += s_loss
                model.writer.add_summary(Sum, step)
                step += 1
                print('batch {:4d}/{:4d}:acc {:1.5f}, c_loss {:>5.3f}, s_loss {:>5.3f}'.format(batch, 
                                                                                    data.num_batch_train, 
                                                                                    acc,
                                                                                    c_loss, 
                                                                                    s_loss), 
                                                                                    end="\r")
                batch += 1

            print("Epoch {}: c_loss {}, s_loss {}, acc:{}".format(epoch, total_c_loss, total_s_loss, accuracy/data.num_batch_train))
    
            saver.save(sess, 'checkpoints/check_points', global_step = model.global_step, write_meta_graph=False)
            if epoch % 1 == 0:
                accuracy = 0.0
                for x_test, y_test in data.next_batch(mode = 'test'):
                    feed_dict = {model.input_matrix: x_test, model.label_one_hot: y_test}
                    acc = sess.run(model.accuracy, feed_dict = feed_dict)
                    accuracy += acc
                print("Test_acc:{}".format(accuracy/data.num_batch_test))
            
def main():
    size = int(sys.argv[1])
    n_epoch = int(sys.argv[2])
    lambda_ = float(sys.argv[3])
    # data = Dataset(batch_size = 16, folder = '/home/dung/my_project/Pic', size = (size, size))
    data = Dataset(batch_size = 32, folder = '/media/trungdunghoang/4022D29E22D297EC/sample', size = (size, size))
    model = Resnet_center(name='resnet_center', input_shape = (size, size, 3), n_classes = len(data.labels), lambda_ = lambda_)
    model.build()
    train_model(model, n_epoch, data)

if __name__ == '__main__':
    main()

from __future__ import print_function
import tensorflow as tf
import numpy as np 
from dataset import Dataset
import math
import os
import cv2 
import sys
from Resnet import Resnet

class Resnet_softmax(Resnet):
    def __init__(self, name, input_shape, n_classes, momentum = 0.9, n_features = 4096, batch_size = 32, learning_rate = 0.01):
        """
            name: string, name of model using for saving graph
            input_shape: tuple, shape of image
            n_classes: int, number of classes
            momentum: float, use for momentum optimizer
            n_features: int, size of embedded features
            batch_size: int, number of images of a batch
            learning_rate: float, learning rate of opitmizer
        
        """
        Resnet.__init__(self, name = name, 
                    input_shape = input_shape, 
                    n_classes = n_classes, 
                    n_features = n_features, 
                    batch_size = batch_size, 
                    learning_rate = learning_rate)
        self.momentum = momentum

    def _create_loss(self):
        with tf.device("/gpu:0"):
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.label_one_hot * tf.log(self.full3), reduction_indices=[1]))
        tf.summary.scalar("loss", self.loss)
    def _create_optimizer(self):
        with tf.device("/gpu:0"):
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.loss, global_step = self.global_step)

    def _create_evaluater(self):
        correct = tf.equal(tf.argmax(self.full3, 1), tf.argmax(self.label_one_hot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        tf.summary.scalar("accuracy", self.accuracy)
    def build(self):
        self._create_placeholder()
        self._build_model()
        self._create_loss()
        self._create_optimizer()
        self._create_evaluater()
        self.merged_summary = tf.summary.merge_all()

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
        # model.writer.add_graph(sess.graph)
        step = 1
        for epoch in range(n_epoch):
            batch = 0
            total_loss = 0.0
            accuracy = 0.0
            for x_train, y_train in data.next_batch(mode = 'train'):
                feed_dict = {model.input_matrix: x_train, model.label_one_hot: y_train}
                loss, acc, Sum, _  = sess.run([model.loss, model.accuracy, model.merged_summary, model.optimizer], feed_dict = feed_dict)
                accuracy += acc
                total_loss += loss
                model.writer.add_summary(Sum, step)
                step += 1
                print('{}/{}:{}'.format(batch, data.num_batch_train, acc), end="\r")
                batch += 1
            print("Epoch {}: loss {}, acc:{}".format(epoch, total_loss, accuracy/data.num_batch_train))
            # writer.add_summary(summary, global_step = epoch)
            # saver.save(sess, 'checkpoints/check_points', global_step = model.global_step, write_meta_graph=False)
            if epoch % 5 == 0:
                accuracy = 0.0
                for x_test, y_test in data.next_batch(mode = 'test'):
                    feed_dict = {model.input_matrix: x_test, model.label_one_hot: y_test}
                    # loss, _, summary  = sess.run([model.loss, model.optimizer, model.summary_op],
                    #                             feed_dict = feed_dict)
                    acc = sess.run(model.accuracy, feed_dict = feed_dict)
                    accuracy += acc
                print("Test_acc:{}".format(accuracy/data.num_batch_test))
            
def main():
    size = int(sys.argv[1])
    n_epoch = int(sys.argv[2])
    # data = Dataset(batch_size = 32, folder = '/mnt/3TB-disk3/data/image-processing/face2/aligned_images_DB', size = (size, size))
    data = Dataset(batch_size = 32, folder = '../../sample', size = (size, size))
    model = Resnet_softmax(name = 'resnet_softmax', input_shape = (size, size, 3), n_classes = len(data.labels))
    model.build()
    train_model(model, n_epoch, data)

if __name__ == '__main__':
    main()

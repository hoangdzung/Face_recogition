"""
    Structure of a batch: A batch includes 3 parts and has an attribute: n_triplet (int)
    The first n_triplet images is anchor part.
    The next n_triplet images is possitive part.
    The last n_triplet images is negative part.
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np 
from triplet_dataset import Dataset
import math
import os
import cv2 
import sys
from Resnet import Resnet

class Resnet_triplet(Resnet):
    def __init__(self, name, input_shape, n_classes, margin = 0.5, threshold = 0.5, momentum = 0.9, n_features = 4096, batch_size = 32, learning_rate = 1e-3):
        """
            name: string, name of model using for saving graph
            input_shape: tuple, shape of image
            n_classes: int, number of classes
            margin: float, margin between negative and possitive distance
            threshold: float, threshold for deciding 2 images are in the same class or not
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
        self.margin = margin
        self.threshold = threshold
        self.momentum = momentum

    def _create_loss(self):
        anchor, pos, neg = tf.split(self.embedding, 3)
        
        aps = tf.reduce_sum(tf.square(anchor - pos))
        ans = tf.reduce_sum(tf.square(anchor - neg))
        
        self.loss = tf.maximum(0., self.batch_size * self.margin + aps - ans)
        
    def _create_optimizer(self):
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, global_step = self.global_step)

    def _create_evaluater(self):
        p_same = 1e-5
        p_diff = 1e-5
        TA = 0.0
        FA = 0.0
        self.label = tf.argmax(self.label_one_hot, axis = 1)
        for i in range(self.batch_size):
            for j in range(i + 1, self.batch_size):
                predict_same = tf.less(tf.norm(self.embedding[i] - self.embedding[j]), self.threshold)
                if tf.equal(self.label[i], self.label[j]) is True:
                    p_same += 1
                    if predict_same is True:
                        TA += 1
                else:
                    p_diff += 1
                    if predict_same is True:
                        FA += 1
        self.VAL = tf.divide(FA, p_same)
        self.FAR = tf.divide(TA, p_diff)

    def build(self):
        self._create_placeholder()
        self._build_model()
        self._create_loss()
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
        # model.writer.add_graph(sess.graph)
        step = 1
        for epoch in range(n_epoch):
            batch = 0
            total_loss = 0.0
            # Val = 0.0
            # Far = 0.0
            P_s = P_d = Ta = Fa = 0.0
            for x_train, y_train in data.next_batch(mode = 'train'):
                feed_dict = {model.input_matrix: x_train, model.label_one_hot: y_train}

                # loss, _, summary  = sess.run([model.loss, model.optimizer, model.summary_op],
                #                             feed_dict = feed_dict)
                loss, p_s, p_d, ta, fa , _  = sess.run([model.loss, model.p_same, model.p_diff, model.TA, model.FA, model.optimizer], feed_dict = feed_dict)
                # Val += val
                # Far += far
                P_s += p_s
                P_d += p_d
                Ta += ta
                Fa += fa
                total_loss += loss
                # model.writer.add_summary(Sum, step)
                step += 1
                print('{}/{}:{},{},{},{}'.format(batch, data.num_batch_train, ta, p_s, fa, p_d), end="\r")
                batch += 1
            # print("Epoch {}: loss {}, val:{}, far: {}".format(epoch, total_loss, Val/data.num_batch_train, Far/data.num_batch_train))
            print("Epoch {}: loss {}, Ta:{}, Ps: {}, Fa:{}, Pd: {}".format(epoch, total_loss, Ta, P_s, Fa, P_d))
            # saver.save(sess, 'checkpoints/check_points', global_step = model.global_step, write_meta_graph=False)
            if epoch % 2 == 1:
                Val = 0.0
                Far = 0.0
                for x_test, y_test in data.next_batch(mode = 'test'):
                    feed_dict = {model.input_matrix: x_test, model.label_one_hot: y_test}
                    val, far = sess.run([model.VAL, model.FAR], feed_dict = feed_dict)
                    Val += val
                    Far += far
                print("val:{}, far:{}".format(Val/data.num_batch_test, Far/data.num_batch_test))
            
def main():
    size = int(sys.argv[1])
    n_epoch = int(sys.argv[2])
    margin = float(sys.argv[3])
    threshold = float(sys.argv[4])
    # data = Dataset(batch_size = 32, folder = '/mnt/3TB-disk3/data/image-processing/face2/aligned_images_DB', size = (size, size))
    data = Dataset(n_triplet = 20, folder = '/home/dung/my_project/cele', size = (size, size))
    model = Resnet_triplet(name = 'Resnet_triplet_2', input_shape = (size, size, 3), n_classes = data.n_classes, batch_size = data.batch_size, margin = margin, threshold = threshold)
    train_model(model, n_epoch, data)

if __name__ == '__main__':
    main()
"""
    Generating a batch is random. (module: dataset)
    For each image in a batch, using L2_distance to find hard pos and hard neg.
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np 
from dataset import Dataset
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

    def cvt_to_dist_matrix(self, embedding):
        """
            Input: 
                n embeding, shape = [n, n_features]
            Output:
                nxn distance matrix of n embedding
        """
        qexpand = tf.expand_dims(embedding, 1)                                                                                                         
        qTexpand = tf.expand_dims(embedding, 0)                                                                                                           
        qtile = tf.tile(qexpand,[1, self.batch_size, 1])                                                                                                                   
        qTtile = tf.tile(qTexpand, [self.batch_size, 1, 1])                                                                                                                 
        deltaQ = qtile - qTtile                                                                                                                            
        deltaQ2 = deltaQ * deltaQ                                                                                                                            
        d2Q = tf.reduce_sum(deltaQ2, 2)  
        d2Q_ = tf.sqrt(d2Q)
        return d2Q_
    
    def _index_max_min(self, dist_matrix, i):
        """
            Input: nxn distance matrix
            Output:
                index of hard negative and hard possitive
        """

        pos_mask = tf.equal(self.label[i], self.label)
        pos_index = tf.where(pos_mask)
        neg_mask = tf.logical_not(pos_mask)
        neg_index = tf.where(neg_mask)

        pos_embed = tf.gather(dist_matrix[i], pos_index)
        neg_embed = tf.gather(dist_matrix[i], neg_index)

        arg_max_in_pos = tf.cast(tf.squeeze(tf.argmax(pos_embed, axis = 0)), tf.int32)
        arg_min_in_neg = tf.cast(tf.squeeze(tf.argmin(neg_embed, axis = 0)), tf.int32)
        
        arg_pos = pos_index[arg_max_in_pos]
        arg_neg = neg_index[arg_min_in_neg]
        
        return arg_pos, arg_neg

    def _create_loss(self):
        dist_matrix = self.cvt_to_dist_matrix(self.full2)
        self.label = tf.argmax(self.label_one_hot, axis = 1)
        pos = [] 
        neg = []
        for i in range(self.batch_size):
            arg_pos, arg_neg = self._index_max_min(dist_matrix, i)
            pos.append(arg_pos)
            neg.append(arg_neg)
        
        anchor = self.embedding
        positive = tf.gather(anchor, pos)
        negative = tf.gather(anchor, neg)
        
        aps = tf.reduce_sum(tf.square(anchor - positive))
        ans = tf.reduce_sum(tf.square(anchor - negative))
        
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
    data = Dataset(batch_size = 64, folder = '/home/dung/my_project/cele', size = (size, size))
    model = Resnet_triplet(input_shape = (size, size, 3), n_classes = len(data.labels), batch_size = data.batch_size, margin = margin, threshold = threshold)
    train_model(model, n_epoch, data)

if __name__ == '__main__':
    main()

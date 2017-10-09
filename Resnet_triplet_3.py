from __future__ import print_function
import tensorflow as tf
import numpy as np 
from triplet_dataset_2 import Dataset
import math
import os
import cv2 
import sys
from Resnet import Resnet

class Resnet_triplet(Resnet):
    def __init__(self, name, input_shape, n_classes, people_per_batch, images_per_people, margin = 0.5, threshold = 0.5, momentum = 0.9, n_features = 4096, batch_size = 32, learning_rate = 1e-3):
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
        self.people_per_batch = people_per_batch
        self.images_per_people = images_per_people
        self.margin = margin
        self.threshold = threshold
        self.momentum = momentum

    def _create_loss(self):
        anchor, pos, neg = tf.split(self.embedding, 3)
        
        aps = tf.reduce_sum(tf.square(anchor - pos))
        ans = tf.reduce_sum(tf.square(anchor - neg))
        
        self.loss = tf.maximum(0., self.batch_size * self.margin + aps - ans)
        tf.summary.scalar("loss", self.loss)

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
        tf.summary.scalar("val", self.VAL)
        tf.summary.scalar("far", self.FAR)
    
    def evaluate(self, embedding, label_one_hot):
        p_same = 1e-5
        p_diff = 1e-5
        TA = 0.0
        FA = 0.0
        label = np.argmax(label_one_hot, axis = 1)
        for i in range(self.batch_size):
            for j in range(i + 1, self.batch_size):
                predict_same = np.linalg.norm(embedding[i] - embedding[j]) < self.threshold
                if label[i] == label[j]:
                    p_same += 1.0
                    if predict_same:
                        TA += 1.0
                else:
                    p_diff += 1.0
                    if predict_same:
                        FA += 1.0
        VAL = FA /p_same
        FAR = TA /p_diff
        return VAL, FAR

    def build(self):
        self._create_placeholder()
        self._build_model()
        self._create_loss()
        self._create_optimizer()
        # self._create_evaluater()
        self.merged_summary = tf.summary.merge_all()

    def triplet_selection(self, embedding):
        emb_start_idx = 0
        a = []
        p = []
        n = []

        for class_label in range(self.people_per_batch):
            for i in range(0, self.images_per_people):
                a_idx = emb_start_idx + i
                neg_dist = np.sum(np.square(embedding[a_idx] - embedding), 1)
                p_idx = np.argmax(neg_dist[emb_start_idx : emb_start_idx + self.images_per_people])
                pos_dist = np.sum(np.square(embedding[a_idx] - embedding[p_idx]))

                neg_dist[emb_start_idx : emb_start_idx + self.images_per_people] = np.inf
                n_idx = np.argmin(neg_dist)
                
                a.append(a_idx)
                p.append(p_idx)
                n.append(n_idx)

            emb_start_idx += self.images_per_people

        index = a + p + n
        return index

def train_model(model, n_epoch, data):
    # file_ckp = "checkpoints_" + model.name
    # saver = tf.train.Saver()
    # if not os.path.exists(file_ckp):
    #     os.mkdir(file_ckp)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        # ckpt = tf.train.latest_checkpoint(file_ckp)
        # print(ckpt)
        # if ckpt:
        #     saver.restore(sess, ckpt)
        model.writer.add_graph(sess.graph)
        step = 1
        for epoch in range(n_epoch):
            batch = 0
            total_loss = 0.0
            for x_train, y_train in data.next_batch(mode = 'train'):
                feed_dict = {model.input_matrix: x_train}
                embedding_matrix = sess.run(model.embedding, feed_dict = feed_dict)
                triplet_index = model.triplet_selection(embedding_matrix)
                # triplet_index_ = [np.arange(i*model.n_features, (i+1)*model.n_features) for i in triplet_index]
                # triplet_embedding = np.take(embedding_matrix, triplet_index_)
                # loss, _, Sum = sess.run([model.loss, model.optimizer, model.merged_summary], feed_dict = {model.embedding_: triplet_embedding})

                triplet_input = np.array([x_train[i] for i in triplet_index])
                loss, _, Sum = sess.run([model.loss, model.optimizer, model.merged_summary], feed_dict = {model.input_matrix: triplet_input})
                total_loss += loss
                model.writer.add_summary(Sum, step)
                step += 1
                print('{}/{}: loss {}'.format(batch, data.num_batch_train, loss), end="\r")
                batch += 1
        
            print("Epoch {}: loss {}".format(epoch, total_loss))
            # saver.save(sess, 'checkpoints/check_points', global_step = model.global_step, write_meta_graph=False)
            if epoch % 3 == 2:
                Val = 0.0
                Far = 0.0
                for x_test, y_test in data.next_batch(mode = 'test'):
                    feed_dict = {model.input_matrix: x_test, model.label_one_hot: y_test}
                    embedding = sess.run(model.embedding, feed_dict = feed_dict)
                    print(embedding.shape)
                    val, far = model.evaluate(embedding, y_test)
                    Val += val
                    Far += far
                print("val:{}, far:{}".format(Val/data.num_batch_test, Far/data.num_batch_test))
                
def main():
    size = int(sys.argv[1])
    n_epoch = int(sys.argv[2])
    margin = float(sys.argv[3])
    threshold = float(sys.argv[4])

    # data = Dataset(batch_size = 32, folder = '/mnt/3TB-disk3/data/image-processing/face2/aligned_images_DB', size = (size, size))
    data = Dataset(people_per_batch = 4, images_per_person = 4, folder = '../../sample', size = (size, size))
    model = Resnet_triplet(name = 'resnet_triplet_3', 
                        input_shape = (size, size, 3), 
                        n_classes = data.n_classes, 
                        people_per_batch = data.people_per_batch, 
                        images_per_people = data.images_per_person,
                        margin = margin, 
                        threshold = threshold)
    model.build()
    train_model(model, n_epoch, data)

if __name__ == '__main__':
    main()

"""
    Everytime sample a batch, choose randomly a class for anchor label.
    Then, choose randomly possitive entities of the chosen class.
    Next, choose randomly negative labels which is different from anchor label.
    Finally, for each negative label, choose randomly an entity.
"""

import numpy as np 
# import cv2 
import os 
import math
import tensorflow as tf

class Dataset():
    def __init__(self, n_triplet, folder, size = (224, 224)):
        self.n_triplet = n_triplet
        self.batch_size = 3 *self.n_triplet
        self.folder = folder
        self.labels = tuple(os.listdir(self.folder))
        self.n_classes = len(self.labels)
        self._load_data()
        self.size = size 

    def _load_data(self):
        self.path_dict_train = {}
        self.path_dict_test = {}
        for i, label in enumerate(self.labels):
            list_img_path = []
            for root, _, files in os.walk(os.path.join(self.folder, label)):
                for file_ in files:
                    list_img_path.append(os.path.join(root, file_))
            n_img = len(list_img_path)
            n_train = int(n_img * 0.8)
            # n_test = n_img - n_train 
            self.path_dict_train[i] = list_img_path[:n_train]
            self.path_dict_test[i] = list_img_path[n_train:]
        
    def next_batch(self, mode = 'train'):
        idx = 0
        start = 0
        if mode == 'train':
            img_dict = self.path_dict_train
        elif mode == 'test':
            img_dict = self.path_dict_test
        while(True):
            anchor_label = np.random.randint(self.n_classes, size = (self.n_triplet))
            a =[]
            for _ in range(self.n_triplet):
            a = [range(self.n_classes)] * self.n_triplet
            for i, j in enumerate(a):
                j.remove(anchor_label[i])
            negative_label = [np.random.choice(np.array(i)) for i in a]
            
            anchor_sample_idx = [np.random.randint(len(img_dict[i])) for i in anchor_label]
            pos_sample_idx = [np.random.randint(len(img_dict[i])) for i in anchor_label]
            neg_sample_idx = [np.random.randint(len(img_dict[i])) for i in negative_label]
            print(anchor_label, negative_label)
            images_path = []
            for label, idx in zip(anchor_label, anchor_sample_idx):
                images_path.append(img_dict[label][idx])

            for label, idx in zip(anchor_label, pos_sample_idx):
                images_path.append(img_dict[label][idx])
            
            for label, idx in zip(negative_label, neg_sample_idx):
                images_path.append(img_dict[label][idx])

            images = []
            labels = []
            for path in images_path:
                image = cv2.imread(path)
                image = cv2.resize(image, self.size)
                images.append(image)
                
                name = path.split('/')[-2]
                label = [1 if name == i else 0 for i in self.labels]
                labels.append(label)
            idx += 1
            start += self.batch_size
            images = np.array(images)
            labels = np.array(labels)
            yield (images, labels)

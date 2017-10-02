"""
    Everytime sample a batch, pick "images_per_person" images per person. 
    And we have "peope_per_batch" people per batch.
"""


import numpy as np 
import cv2 
import os 
import math
import tensorflow as tf

class Dataset():
    def __init__(self, people_per_batch, images_per_person, folder, size = (224, 224)):
        self.people_per_batch = people_per_batch
        self.images_per_person = images_per_person
        # self.batch_size = 3 *self.n_triplet
        self.folder = folder
        self.labels = tuple(os.listdir(self.folder))
        self.n_classes = len(self.labels)
        self._load_data()
        self.size = size 

    def _load_data(self):
        self.path_dict_train = {}
        self.n_people_per_class_train = []
        self.path_dict_test = {}
        self.n_people_per_class_test = []
        for i, label in enumerate(self.labels):
            list_img_path = []
            for root, _, files in os.walk(os.path.join(self.folder, label)):
                for file_ in files:
                    list_img_path.append(os.path.join(root, file_))
            n_img = len(list_img_path)
            n_train = int(n_img * 0.8)
            n_test = n_img - n_train 
            self.path_dict_train[i] = list_img_path[:n_train]
            self.n_people_per_class_train.append(n_train)
            self.path_dict_test[i] = list_img_path[n_train:]
            self.n_people_per_class_test.append(n_test)
        self.images_per_person = min(self.images_per_person, min(self.n_people_per_class_train))
        self.people_per_batch = min(self.people_per_batch, self.n_classes)
        self.batch_size = self.images_per_person * self.people_per_batch
        
    @staticmethod
    def random_choice(m, n):
        """
            Choose m integer numbers from range(n)
        """
        assert m < n
        all_label = np.arange(n)
        np.random.shuffle(all_label)
        choosen_label = all_label[:m]
        return choosen_label

    def next_batch(self, mode = 'train'):
        if mode == 'train':
            img_dict = self.path_dict_train
            n_people_per_class = self.n_people_per_class_train
        elif mode == 'test':
            img_dict = self.path_dict_test
            n_people_per_class = self.n_people_per_class_test

        while(True):
            anchor_class_ = self.random_choice(self.people_per_batch, self.n_classes)
            anchor_sample_idx = [self.random_choice(self.images_per_person, n_people_per_class) for i in anchor_class_]
            
            images = []
            labels = []

            for class_, idx in zip(anchor_class_, anchor_sample_idx):
                image = cv2.imread(img_dict[class_][idx])
                image = cv2.resize(image, self.size)
                images.append(image)
                label = np.zeros(self.n_classes)
                label[class_] = 1
                labels.append(label)

            images = np.array(images)
            labels = np.array(labels)
            yield (images, labels)

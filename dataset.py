import numpy as np 
import cv2 
import os 
import math
import tensorflow as tf

class Dataset():
    def __init__(self, batch_size, folder, size = (250, 250)):
        self.batch_size = batch_size
        self.folder = folder
        self.labels = tuple(os.listdir(self.folder))
        self._load_data()
        self.size = size 

    def _load_data(self):
        self.path_list = []
        for root, dirs, files in os.walk(self.folder):
            for i in files:
                self.path_list.append(os.path.join(root, i))
        np.random.shuffle(self.path_list)

        self.num_batch = len(self.path_list)/self.batch_size
        self.len_data = self.num_batch * self.batch_size
        self.path_list = self.path_list[:self.len_data]
        self.num_batch_train = int(0.8*self.num_batch)
        self.num_batch_test = self.num_batch - self.num_batch_train
        self.len_train = self.num_batch_train * self.batch_size
        self.len_test = self.num_batch_test *self.batch_size
        # self.len_data = len(self.path_list)
        # self.len_train = int(0.8*self.len_data)
        # self.len_test = self.len_data - self.len_train
        # self.num_batch_train = int(math.ceil(float(self.len_train) / self.batch_size))
        # self.num_batch_test = int(math.ceil(float(self.len_test) / self.batch_size))

        self.train = self.path_list[:self.len_train]
        self.test = self.path_list[self.len_train:]

    def next_batch(self, mode = 'train'):
        idx = 0
        start = 0
        if mode == 'train':
            np.random.shuffle(self.train)
            while(idx < self.num_batch_train):
                images_path = self.train[start : start + self.batch_size]
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
        
        elif mode == 'test':
            while(idx < self.num_batch_test):
                images_path = self.test[start : start + self.batch_size]
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

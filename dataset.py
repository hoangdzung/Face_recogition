import numpy as np 
import cv2 
import os 
import math
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, wait
import time
from threading import Thread

class Dataset():
    def __init__(self, batch_size, folder, size = (250, 250)):
        self.batch_size = batch_size
        self.folder = folder
        self.labels = tuple(os.listdir(self.folder))
        self._load_data()
        self.size = size 
        self.n_classes = len(self.labels)

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

    def next_batch_2(self, mode = 'train', batch_size = None):
        idx = 0
        start = 0
        if mode == 'train':
            data = self.train
            num_batch = self.num_batch_train
        elif mode == 'test':
            data = self.test
            num_batch = self.num_batch_test

        if batch_size is None:
            batch_size = self.batch_size
        else:
            num_batch = int(math.ceil((float)(len(data))/batch_size))
        
        np.random.shuffle(data)
        while(idx < num_batch):
            images_path = data[start : start + batch_size]
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
            start += batch_size
            images = np.array(images)
            labels = np.array(labels)
            yield (images, labels)
    
    def prepare_data(self, images_path):
        images = []
        labels = []
        for path in images_path:
            image = cv2.imread(path)
            image = cv2.resize(image, self.size)
            images.append(image)
            
            name = path.split('/')[-3]
            label = [1 if name == i else 0 for i in self.labels]
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)
        return (images, labels)

    def next_batch(self, mode = 'train', batch_size = None):
        if mode == 'train':
            data = self.train
            num_batch = self.num_batch_train
        elif mode == 'test':
            data = self.test
            num_batch = self.num_batch_test

        if batch_size is None:
            batch_size = self.batch_size
        else:
            num_batch = int(math.ceil((float)(len(data))/batch_size))

        np.random.shuffle(data)
        idx = 0
        start = 0
        pool = ThreadPoolExecutor(1)
        future = pool.submit(self.prepare_data, data[start:start+batch_size])
        start += batch_size
        while(idx < num_batch - 1):
            wait([future])
            minibatch = future.result()
            # While the current minibatch is being consumed, prepare the next
            future = pool.submit(self.prepare_data, data[start:start+batch_size])
            yield minibatch
            idx += 1
            start += batch_size
        # Wait on the last minibatch
        wait([future])
        minibatch = future.result()
        yield minibatch

from triplet_dataset_2 import Dataset
import numpy as np

data = Dataset(people_per_batch = 3, images_per_person = 4, folder = '/home/trungdunghoang/Desktop/A', size = (200, 200))
# data = Dataset(batch_size = 32, folder = '/home/trungdunghoang/Desktop/A', size = (200, 200))
i=0
for x,y in data.next_batch():
    print(x.shape, y.shape)
# from keras.applications.resnet50 import ResNet50
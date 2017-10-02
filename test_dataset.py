from dataset import Dataset
import numpy as np

data = Dataset(batch_size = 32, folder = '/home/trungdunghoang/Desktop/A', size = (200, 200))
for x,y in data.next_batch():
    print(x.shape, y.shape)
# from keras.applications.resnet50 import ResNet50
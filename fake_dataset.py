import numpy as np 

class Fake_data():
    def __init__(self, batch_size, n_classes, size):
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.size = size 
        self._create_dataset()
    
    def _create_dataset(self):
        self.X = []
        self.Y = []
        for cls_ in range(self.n_classes):
            for _ in range(50):
                x = 0.1 * np.random.randn(self.size[0], self.size[1], 3) + cls_
                y = np.zeros(self.n_classes)
                y[cls_] = 1

                self.X.append(x)
                self.Y.append(y)
        self.data = list(zip(self.X, self.Y))
        self.len_data = len(self.X)
        self.num_batch = self.len_data // self.batch_size

    def next_batch_train(self):
        idx = 0
        start = 0
        np.random.shuffle(self.data)
        while(idx < self.num_batch):
            data = self.data[start : start + self.batch_size]
            images = []
            labels = []
            for i in data:
                image, label = i
                images.append(image)
                labels.append(label)
            idx += 1
            start += self.batch_size
            images = np.array(images)
            labels = np.array(labels)
            yield (images, labels)
        
    def next_batch_test(self, size = 2):
        X = []
        Y = []
        labels = np.random.randint(self.n_classes, size = size)
        for label in labels:
            x = 0.01 * np.random.randn(self.size[0], self.size[1], 3) + label
            y = np.zeros(self.n_classes)
            y[label] = 1

            X.append(x)
            Y.append(y)
        return np.array(X), np.array(Y)
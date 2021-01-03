import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def data_loader():
    
    file_name = '/home/wilfred/Datasets/cifar-10-python/cifar-10-batches-py/data_batch_1'

    bt_1 = unpickle(file_name)

    y = bt_1[b'labels']
    X = bt_1[b'data']

    X_new = np.zeros(shape=[X.shape[0], 32, 32, 3], dtype=np.uint8)

    for i in range(X.shape[0]):
        im_r = X[i][0:1024].reshape(32, 32)
        im_g = X[i][1024:2048].reshape(32, 32)
        im_b = X[i][2048:].reshape(32, 32)
        img = np.dstack((im_r, im_g, im_b))
        X_new[i] = img
    
    return X_new[9999]

if __name__ == '__main__':

    img = data_loader()
    plt.imshow(img)
    plt.show()

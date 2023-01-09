from perceptron import *
from mnist import MNIST

import matplotlib.pyplot as plt
import numpy as np
import os


def load_data():
    data = MNIST(os.getcwd())

    images, labels = data.load_training()

    print(f'loaded training images and labels')

    return images, labels

def show_img(images, labels, index):
    width = int(np.sqrt(len(images[index])))

    plt.imshow(np.array(images[index]).reshape(width, width), cmap="gray", vmin = 0, vmax = 255)

    plt.title(f'size : {width}x{width}, label : {labels[index]}')

    plt.show()

imgs,labels = load_data()


net = init_network([784, 30,10])
print(f'network initialized')

# goals
iterations = 100000
error      = 0.01

i = 0
err = 1

batch = slice(200,230, 1)

img_batch = np.array(imgs[batch]).T.astype(float)/256
lbl_batch = labels[batch]
lbl_img_batch = encode(lbl_batch)


while i <iterations and err >error:
    i+=1
    net, err = Update(*net, img_batch, lbl_img_batch, .001)
    print(f"iteration {i}, error {err}")


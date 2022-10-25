#imports
from traceback import format_exc
from mnist import MNIST
import os
import matplotlib.pyplot as plt
import numpy as np


def sigma(x):
    if x < 0.:
        return np.exp(x) / (1.+np.exp(x))
    else:
        return 1./(1. + np.exp(-x))

sigmoid = np.vectorize(sigma)

def sigmoid_prime(x):

    return np.exp(-np.abs(x)) / (1. + np.exp(-np.abs(x)))

#intializing MNIST
data = MNIST(os.getcwd())

#loading train_images and labels for them
#train_images, train_labels = data.load_training()
#loading test_images and labels for them
test_images, test_labels = data.load_testing()

#shows image with given index of givrn type train/test
def show_image(type:str, index:int):
    '''
    if type.lower() == "train":

        plt.imshow(np.array(train_images[index]).reshape(28,28), cmap= 'gray')
        plt.title(f"this is {train_labels[index]}")
        plt.show()

    elif type.lower() == "test":
    '''
    plt.imshow(np.array(test_images[index]).reshape(28,28), cmap = "gray", vmin = 0, vmax = 255)
    plt.title(f"this is {test_labels[index]}")
    plt.show()
    '''
    else:
        return "incorrect type only train/test are valid"
    '''

#   example of use show_image()
#       show_image("test", 200)

sizes = [784, 30, 10]
num_layers = len(sizes)

batch_size = 100

biases  = [np.random.randn(y,1 ) for y in sizes[1:]]
weights = np.array([np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])])

def feedforward(a):
    for bias, weight in zip(biases, weights):
        a = sigmoid(np.dot(weight, a) + bias)

    return a

u = feedforward(np.array([test_images[i] for i in range(300,300+batch_size)]).reshape(784, batch_size))
print(u.shape)

show_image('test',300)


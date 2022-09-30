#imports
from mnist import MNIST
import os
import matplotlib.pyplot as plt
import numpy as np

#intializing MNIST
data = MNIST(os.getcwd())

#loading train_images and labels for them
train_images, train_labels = data.load_training()
#loading test_images and labels for them
test_images, test_labels = data.load_testing()

#shows image with given index of givrn type train/test
def show_image(type:str, index:int):
    
    if type.lower() == "train":

        plt.imshow(np.array(train_images[index]).reshape(28,28))
        plt.title(f"this is {train_labels[index]}")
        plt.show()

    elif type.lower() == "test":

        plt.imshow(np.array(test_images[index]).reshape(28,28))
        plt.title(f"this is {test_labels[index]}")
        plt.show()

    else:
        return "incorrect type only train/test are valid"


#   example of use show_image()
#       show_image("train", 200)
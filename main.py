from perceptron import *
from mnist import MNIST

import matplotlib.pyplot as plt
import numpy as np
import os


def load_data():
    data = MNIST(os.getcwd())

    images, labels = data.load_training()

    print(f'loaded {len(images)} training images and labels')

    test_images, test_labels = data.load_testing()

    print(f'loaded {len(test_images)} testing images and labels')
    return images, labels, test_images, test_labels

def show_img(images, labels, index):
    width = int(np.sqrt(len(images[index])))



    plt.imshow(np.array(images[index]).reshape(width, width), cmap="gray", vmin = 0, vmax = 255)

    plt.title(f'size : {width}x{width}, label : {labels[index]}')

    plt.show()

imgs,labels, test_images, test_labels = load_data()


net = init_network([784, 30,10])
print(f'network initialized')

# goals
iterations = 1000000
error      = 0.001

i = 0
err = 1

batch = [slice(i,i+70, 1) for i in range(0, 60000, 70)]


img_batch = [np.array(imgs[i]).T.astype(float)/256 for i in batch]
lbl_batch = [labels[i] for i in batch]
lbl_img_batch = [encode(i) for i in lbl_batch]

b = [slice(i,i+100, 1) for i in range(0, 10000, 100)]

test_img_batch = [np.array(test_images[i]).T.astype(float)/256 for i in b]
test_lbl_batch = [test_labels[i] for i in b]



def GetAccuracy(biases, weights):

    correct_sum = 0
    x = 0
    for img_b, lbl_b in zip(test_img_batch, test_lbl_batch):
        
        for ans, lbl in zip(forward(biases, weights, img_b).T, lbl_b):
            #print(forward(biases, weights, img).shape)
            #print(decode(i), lbl[num])
            #print(i,lbl[:,num] , decode(i), decode(lbl[:,num]),  decode(i) == decode(lbl[:,num]))
            if decode(ans) == lbl:
                
                correct_sum+=1
            x+=1 
    
    print(correct_sum, x)
    return correct_sum / x *100

    
while i <iterations and err >error:
    i+=1
    es = 0
    for img, lbl in zip(img_batch, lbl_img_batch):
        net, err = Update(*net, img, lbl, .001)
        es+=err
    print(f"iteration {i}, error {es/len(img_batch)}", end = " ")
    print(f"iteration accuracy: {GetAccuracy(*net)}%")


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

def cost_func(output, labels):
    return [0.5*(labels[i] - output[i]).mean()**2 for i in range(output.shape[0])]

def cost_derivative(output, labels):
    return output - labels


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

class Network():
    def __init__(self, sizes, batch, data, learning_rate = 0.1):
        self.sizes = sizes #[784, 30, 10]
        self.num_layers = len(self.sizes)

        self.batch_size = batch # = 100

        self.learning_rate = learning_rate

        self.biases  = [np.random.randn(y,1 ) for y in sizes[1:]]
        self.weights = np.array([np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])])

        self.activtions    = []
        self.weightedinput = []
        
        self.data   = data

    def feedforward(self ):
        a = self.data
        for bias, weight in zip(self.biases, self.weights):

            z = np.dot(weight, a) + bias
            self.weightedinput.append(z)
            a = sigmoid(z)
            self.activtions.append(a)
        
        
        return a

    def backpropagate(self):
        #INTIAL ERROR
        error = cost_derivative(self.feedforward(), np.array([np.insert(np.zeros(9), test_labels[i], 1) for i in range(300,300+100)]).reshape(10, 100))
        
        bias_gradient   = [np.zeros((len(bias),self.batch_size)) for bias in self.biases] #CREATING LIST FOR BIAS GRADIENTS
        weight_gradient = [np.zeros((len(weight),self.batch_size)) for weight in self.weights] #CREATING LIST FOR WEIGHT GRADIENTS


        bias_gradient[-1]   = error #LAST LAYER 
        weight_gradient[-1] = np.dot(error, self.activtions[-2].T) #LAST LAYER 

        for layer in range(2, self.num_layers):
            weighted_input = self.weightedinput[-layer] # GET WEIGHTED INPUT AT GIVEN LAYER

            sigma_prime = sigmoid_prime(weighted_input) # GET SIGMOID_PRIME AT GIVEN LAYER

            error = np.dot(self.weights[-layer + 1].transpose(),error ) * sigma_prime #CALCULATE ERROR AT GIVEN LAYER

            bias_gradient[-layer] = error
            weight_gradient[-layer] = np.dot(sigma_prime, self.activtions[-layer-1].T)

        return bias_gradient, weight_gradient


model = Network([784,30,10],100, np.array([test_images[i] for i in range(300,300+100)]).reshape(784, 100),  100)
model.backpropagate()


#show_image('test',300)


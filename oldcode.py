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
    #label[i] is a vector!
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
    def __init__(self, sizes):
        self.sizes = sizes #[784, 30, 10]
        
        #self.num_layers = len(self.sizes)

        #self.batch_size = data.shape[-1] # = 100

        #self.learning_rate = learning_rate

        self.biases  = [np.random.randn(y,1 ) for y in sizes[1:]]
        
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
        
        self.activtions    = []
        self.weightedinput = []
        


        #self.data   = data
        #self.y = np.array([np.insert(np.zeros(9), i, 1) for i in labels]).T

    def _toarray(self, labels):
        
        return np.array([np.insert(np.zeros(9), i, 1) for i in labels]).T

    def train(self, data, labels, learning_rate=0.01):
        if data.shape[-1] != labels.shape[-1]:
            raise Exception("not matching labels and data")

        self.num_layers = len(self.sizes)

        self.batch_size = data.shape[-1] # = 100

        self.learning_rate = learning_rate

        self.data   = data
        self.y = self._toarray(labels)

        self._update()

    def _update(self):
        bias_gr, weigth_gr = self.backpropagate()
        n_layers = len(weigth_gr)
        for layer in range(n_layers):

            self.biases[layer]  -= np.array(bias_gr[layer].mean(axis = -1)).reshape(self.sizes[layer+1],1)*self.learning_rate

        print(weigth_gr[-1].mean(axis = -1).shape)
        self.weights = [self.weights[i] - weigth_gr[i].mean(axis=-1)*self.learning_rate for i in range(n_layers) ]

    def feedforward(self,x):
        a = x
        
        for bias, weight in zip(self.biases, self.weights):

            z = np.dot(weight, a) + bias
            self.weightedinput.append(z)
            a = sigmoid(z)
            self.activtions.append(a)
        
        return a

    def backpropagate(self):
        
        error = np.array(cost_derivative(self.feedforward(self.data), self.y))

        bias_gradient   = [np.zeros((len(bias),self.batch_size)) for bias in self.biases] #CREATING LIST FOR BIAS GRADIENTS
        weight_gradient = [np.zeros((weight.shape[0],weight.shape[1], self.batch_size)) for weight in self.weights] #CREATING LIST FOR WEIGHT GRADIENTS

        print(weight_gradient[-1].mean(axis = -1).shape)
        bias_gradient[-1]   = error #LAST LAYER 
        #weight_gradient[-1] = error* self.activtions[-2] #LAST LAYER 
        for i in range(weight_gradient[-1].shape[1]-1):
            for j in range(weight_gradient[-1].shape[0]-1):
                for b in range(self.batch_size):
                    print(f"i:{i}\nj:{j}\nb:{b}")
                    print(weight_gradient[-1].shape)
                    print(self.activtions[-2].shape)
                    weight_gradient[-1][i,j,b] = self.activtions[-2][i,b] * error[j,b]

        for layer in range(2, self.num_layers-1):
            weighted_input = self.weightedinput[-layer] # GET WEIGHTED INPUT AT GIVEN LAYER

            sigma_prime = sigmoid_prime(weighted_input) # GET SIGMOID_PRIME AT GIVEN LAYER

            error = np.dot(self.weights[-layer + 1].transpose(),error ) * sigma_prime #CALCULATE ERROR AT GIVEN LAYER

            bias_gradient[-layer] = error
            weight_gradient[-layer] = np.dot(sigma_prime, self.activtions[-layer-1].T)
        print(weight_gradient[0].shape)
        return bias_gradient, weight_gradient

    def _tonum(self, x):
        return list(x).index(max(list(x)))+1

    def predict(self, data):
        
        return self._tonum(self.feedforward(data).mean(axis=1))



model = Network([784,30,10],)
model.train(np.array(test_images[300:300+100]).T,np.array(test_labels[300:400]), 100)
model.backpropagate()
#
# print(model.predict(np.array(test_images[300]).T, ))

#show_image('test',300)

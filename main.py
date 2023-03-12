
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


imgs, lbls, test_imgs, test_lbls = load_data()
net = init_network([784, 30, 10])

def img_batch(batch): return np.array(imgs[batch]).T.astype(float)/256.
def lbl_batch(batch): return lbls[batch]
def lbl_img_batch(batch): return encode(lbl_batch(batch))

batches = [(img_batch(batch = slice(b, b+70, 1)), lbl_img_batch(slice(b, b+70, 1)), lbl_batch(slice(b, b+70, 1))) for b in range(0, 60000-70, 70)]
# stopping conditions
err_mx = 0.07
cyc_mx = 2000

# initial values of stopping conditions
cyc = 0
err = 0.1 #np.linalg.norm(forward(*net, img_batch) - lbl_img_batch)

# iteration over one batch
while cyc < cyc_mx and err > err_mx:
    for img, lbl, lbl_batch in batches:
        #$print(img, lbl, lbl_batch)
        net, err = Update(*net, model_input=img, labels=lbl, learning_rate=0.3)
        
        error_batch = np.array(lbl_batch) - decode(forward(*net, img))
        rel_error = np.linalg.norm(error_batch, ord = 0)/len(lbl_batch)
        accuracy = 1. - rel_error
    cyc += 1

    print(f"iteration {cyc}, accuracy: {accuracy*100}%")
    
# report results    
if err < err_mx:
    print(f"converged after {cyc} cycles, error: {err}")
else:
    print(f"not converged after {cyc} cycles, error: {err}")

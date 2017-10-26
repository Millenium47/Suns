import random, os
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def FinalShowTrain(pickles):
    fig = plt.figure()
    plt.title('Training data')
    plt.axis('off')
    with open(pickles, 'rb') as f:
        unpickled = pickle.load(f)

        datasets = unpickled.values()
        i = 1
        j = 15

        for image in datasets[0][15:18]:
            a = fig.add_subplot(1,3,i)
            a.title.set_text(datasets[1][j])
            i +=1
            j +=1

            plt.axis('off')
            plt.imshow(image)
    plt.show()


def FinalShowTest(pickles):
    fig = plt.figure()
    plt.title('Testing data')
    plt.axis('off')
    with open(pickles, 'rb') as f:
        unpickled = pickle.load(f)

        datasets = unpickled.values()
        i = 1
        j = 15

        for image in datasets[2][15:18]:
            a = fig.add_subplot(1,3,i)
            a.title.set_text(datasets[3][j])
            i +=1
            j +=1

            plt.axis('off')
            plt.imshow(image)
    plt.show()

def FinalShowValid(pickles):
    fig = plt.figure()
    plt.title('Validation data')
    plt.axis('off')
    with open(pickles, 'rb') as f:
        unpickled = pickle.load(f)

        datasets = unpickled.values()
        i = 1
        j = 15

        for image in datasets[4][15:18]:
            a = fig.add_subplot(1,3,i)
            a.title.set_text(datasets[5][j])
            i +=1
            j +=1

            plt.axis('off')
            plt.imshow(image)
    plt.show()

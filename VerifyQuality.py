import random
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

def verify(datasets):

    totalDatasetCount = 0
    i=1
    fig = plt.figure()

    for dataset in datasets:
        with open(dataset, 'rb') as handle:
            unpickleSet = pickle.load(handle)
            number = random.randint(0,500)
            oneImage = unpickleSet[number,:,:]


            fig.add_subplot(1,10,i)
            i += 1

            plt.axis('off')
            plt.imshow(oneImage)

        totalDatasetCount += unpickleSet.shape[0]
    plt.show()
    return totalDatasetCount
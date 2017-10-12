import os
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle


imageSize = 28
picklesReport = []

#normalize data and convert data to 3D dataset
#input: folderPath= path to folder, minNumber = minimum number of needed images
#output: 3D array of normalized images
def letterToDataset(folderPath):

    images = os.listdir(folderPath)
    count = 0

    #https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.html
    dataset = np.ndarray( shape=(len(images), imageSize, imageSize), dtype=np.float32)

    for image in images:
        images = os.path.join(folderPath, image)

        try:
            normalizedImage = (ndimage.imread(images).astype(float) - 255.0 / 2) / 2 #vzorec z cvika
            if normalizedImage.shape != (imageSize, imageSize):
                raise Exception('Bad shape after normalizing, skipping')

        except IOError as err:
            print('Could not read --> skipping', err)

        dataset[count, :, :] = normalizedImage
        count += 1

    return dataset

#put each dataset to separate letter class
#input: filesPath - path to folder with datas , numberOfImages - min number of images per class
#output: datasets = array with separate datasets
def toPickle(filesPath):
    datasets = []

    for file in filesPath:
        #file format = C:/Users/andrej.duben/PycharmProjects/SUNS1/notMNIST_small/A
        separateSet = file + '.pickle'
        if os.path.isfile(separateSet):
            print('%s already exist --> skipping' %separateSet)
        else:
            dataset = letterToDataset(file)

            try:
                picklesReport.append('Created: %s with image count of %s' %(separateSet, dataset.shape[0]))
                #print('Created: %s with image count of %s' %(separateSet, dataset.shape[0]))
                #https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
                with open(separateSet, 'wb') as handle:
                    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                print('Unable to create pickle %s : %e' %(separateSet, exc))

        datasets.append(separateSet)

    #prints created pickles
    for line in picklesReport:
        print(line)

    return datasets


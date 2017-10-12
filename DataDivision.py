import os, math
import numpy as np
from scipy import ndimage
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt


validation_ratio = 0.15
testing_ratio = 0.15
imageSize = 28

def permutationShuffle(dataset, code):
    permutation = np.random.permutation(code.shape[0])

    shuffledDataset = dataset[permutation,:,:]
    shuffledCode = code[permutation]
    return shuffledDataset, shuffledCode

#input - datasets by letter
#output- datasets by data type
def splitToSets(datasets):

    validation = []
    test = []
    train = []
    validation_code = []
    test_code = []
    train_code = []
    coder = 0

    for dataset in datasets:
        with open(dataset, 'rb') as handle:
            unpickleSet = pickle.load(handle)

            #number of images by ratio from current dataset
            unpickle_test = int(math.floor(testing_ratio*len(unpickleSet)))
            unpickle_validate = int(math.floor(validation_ratio*len(unpickleSet)))
            unpickle_train = len(unpickleSet) - unpickle_test - unpickle_validate

            #load number of images set by ratio to array
            for image in range(0,unpickle_test):
                test.append(unpickleSet[image,:,:])
                test_code.append(coder)
            unpickle_to = unpickle_test + unpickle_validate

            for image in range(unpickle_test,unpickle_to):
                validation.append(unpickleSet[image,:,:])
                validation_code.append(coder)
            unpickle_to += unpickle_train

            for image in range(unpickle_test+unpickle_validate, unpickle_to):
                train.append(unpickleSet[image,:,:])
                train_code.append(coder)
        coder += 1

    #inicialize ndarrays for sets
    validation_dataset = np.ndarray(shape=(len(validation), imageSize, imageSize), dtype=np.float32)
    validation_coder = np.ndarray(shape=(len(validation_code)), dtype=np.int16)

    testing_dataset = np.ndarray(shape=(len(test), imageSize, imageSize), dtype=np.float32)
    testing_coder = np.ndarray(shape=(len(test_code)), dtype=np.int16)

    training_dataset = np.ndarray(shape=(len(train), imageSize, imageSize), dtype=np.float32)
    training_coder = np.ndarray(shape=(len(train_code)), dtype=np.int16)

    #copy arrays to ndarrays
    coder = 0
    for image in validation:
        validation_dataset[coder,:,:] = image
        validation_coder[coder] = validation_code[coder]
        coder += 1

    coder = 0
    for image in test:
        testing_dataset[coder, :, :] = image
        testing_coder[coder] = test_code[coder]
        coder += 1

    coder = 0
    for image in train:
        training_dataset[coder, :, :] = image
        training_coder[coder] = train_code[coder]
        coder += 1

    #shuffle data
    training_dataset, training_coder = permutationShuffle(training_dataset, training_coder)
    testing_dataset, testing_coder = permutationShuffle(testing_dataset, testing_coder)
    validation_dataset, validation_coder = permutationShuffle(validation_dataset, validation_coder)


    print('Total images in training dataset:  %d   %d' %(training_dataset.shape[0], training_coder.shape[0] ))

    print('Total images in testing dataset  %d' %testing_dataset.shape[0])

    print('Total images in validation dataset  %d' %validation_dataset.shape[0])

    print('Total number of images in final pickle file   %d' %(validation_dataset.shape[0] + testing_dataset.shape[0] + training_dataset.shape[0]))

    return training_dataset, validation_coder, testing_dataset, training_coder, validation_dataset, testing_coder

#input - datasets by type
#output - final pickle file
def toOneFile(path, divided):
    path = path + 'Final.pickle'
    if os.path.isfile(path):
        print('%s  already exists' %path)
    else:
        try:
            f = open(path, 'wb')
            save={
                'train_DS'      : divided[0],
                'train_CD'      : divided[1],
                'test_DS'       : divided[2],
                'test_CD'       : divided[3],
                'validatiob_DS' : divided[4],
                'validation_CD' : divided[5],
                }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save final file', e)

    return path
from PIL import Image
import imagehash
import numpy as np


def hashSmall(validation, test, train, trainLabel):
    old_training_dataset = train
    smallHash = set()
    indexToRemove = []

    for image in validation:
        img = Image.fromarray(image, 'RGB')
        smallHash.add(imagehash.average_hash(img))

    for image in test:
        img = Image.fromarray(image, 'RGB')
        smallHash.add(imagehash.average_hash(img))

    i = 0
    for image in train:
        img = Image.fromarray(image, 'RGB')
        hash = imagehash.average_hash(img)

        if smallHash.__contains__(hash):
            indexToRemove.append(i)
        i += 1

    # remove indexes from train datasets and it label array
    training_dataset = np.delete(train, indexToRemove, 0)
    trainig_labels = np.delete(trainLabel, indexToRemove, 0)

    print("Removed images from training data: " + str(old_training_dataset.shape[0] - training_dataset.shape[0]))

    return training_dataset, trainig_labels

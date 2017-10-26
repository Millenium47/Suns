import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

#10 pri 2800 datach
epsSet = 6.95

def DBScanAlghoritm(dataset):
    dbscan = DBSCAN(epsSet, min_samples=30)
    dataset = dataset.reshape(dataset.shape[0], dataset.shape[1] * dataset.shape[2])
    dbscan.fit(dataset)


    print('done clustering')

    cluster=set()

    for label in dbscan.labels_:
        cluster.add(label)
    numberCluster = len(cluster) - 1
    print('Pocet clusterov: ' + str(numberCluster))

    # fill clusters with data in format (i, 28*28)
    dataInClusters = [[] for j in range(0, numberCluster)]
    sum = []
    i = 0
    for label in dbscan.labels_:
        if(label != -1):
            dataInClusters[label].append(dataset[i])
        else:
            sum.append(dataset[i])
        i += 1
    print('Sum ma pocet: '+ str(len(sum)))
    # print sum in cluster and std and save average image per cluster
    averageImgPerCluster = []
    for i in range(0, numberCluster):
        print('Cluster' + str(i) + ' have length: ' + str(len(dataInClusters[i])) + ' and std: ' + str(
            np.std(dataInClusters[i])))
        averageImgPerCluster.append((np.sum(dataInClusters[i], axis=0) / len(dataInClusters[i])))
        plt.imshow(averageImgPerCluster[i].reshape(28, 28))
        plt.subplot(1, numberCluster, i + 1)
        plt.axis('off')
    plt.show()

    # plot nearest image to average image in cluster
    nearestImages = []
    for i in range(0, numberCluster):
        bestDistance = 1000000000000000  # np.finfo(np.float128)
        index = 0
        for j in range(0, len(dataset)):
            # current distance
            distance = calcDistance(dataset[j], averageImgPerCluster[i])  # centers[i]

            if (distance < bestDistance):
                bestDistance = distance
                index = j
        plt.subplot(2, numberCluster, i + 1)

        plt.axis('off')
        plt.imshow(dataset[index].reshape(28, 28))
        print('Best distance for Cluster' + str(i) + ' is: ' + str(bestDistance))

    for k in range(0, 10):
        plt.subplot(2, numberCluster, k + 11)
        plt.axis('off')
        plt.imshow(averageImgPerCluster[k].reshape(28, 28))

    plt.show()


def calcDistance(A, B):
    return np.linalg.norm(A - B)
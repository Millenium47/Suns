import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

clusters = 10


def KmeansAlgorithm(dataset):
    kmeans = KMeans(n_clusters=clusters)
    dataset = dataset.reshape(dataset.shape[0], dataset.shape[1] * dataset.shape[2])
    kmeans.fit(dataset[0:10000])

    print('done clustering')

    # plot center per cluster
    centers = kmeans.cluster_centers_
    i = 1
    for center in centers:
        img = center.reshape(28, 28)

        plt.subplot(1, clusters, i)
        plt.axis('off')
        plt.imshow(img)
        i += 1
    plt.show()

    # fill clusters with data in format (i, 28*28)
    dataInClusters = [[] for j in range(0, clusters)]
    i = 0
    for label in kmeans.labels_:
        dataInClusters[label].append(dataset[i])
        i += 1

    # print sum in cluster and std and save average image per cluster
    averageImgPerCluster = []
    for i in range(0, clusters):
        print('Cluster' + str(i) + ' have length: ' + str(len(dataInClusters[i])) + ' and std: ' + str(
            np.std(dataInClusters[i])))
        averageImgPerCluster.append((np.sum(dataInClusters[i], axis=0) / len(dataInClusters[i])))
        plt.imshow(averageImgPerCluster[i].reshape(28, 28))
        plt.subplot(1, clusters, i + 1)
        plt.axis('off')
    plt.show()

    # plot nearest image to average image in cluster
    nearestImages = []
    for i in range(0, clusters):
        bestDistance = 1000000000000000 # np.finfo(np.float128)
        index = 0
        for j in range(0, len(dataset)):
            #current distance
            distance = calcDistance(dataset[j], averageImgPerCluster[i]) #centers[i]

            if (distance < bestDistance ):

                bestDistance = distance
                index = j
        plt.subplot(2, clusters, i+1)

        plt.axis('off')
        plt.imshow(dataset[index].reshape(28,28))
        print('Best distance for Cluster' +str(i)+ ' is: '+str(bestDistance))

    for k in range(0, 10):
        plt.subplot(2,clusters,k+11)
        plt.axis('off')
        plt.imshow(averageImgPerCluster[k].reshape(28, 28))

    plt.show()

def calcDistance(A, B):
    return np.linalg.norm(A-B)

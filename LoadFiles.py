import os, random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#path = 'C:/Users/andrej.duben/PycharmProjects/SUNS1/notMNIST_small/'

#loads files from path
def load_pictures(path,fig, i):
    fileNames = os.listdir(path)
    random.shuffle(fileNames)

    for image in fileNames[:3]:
        fig.add_subplot(10,3,i)
        firstImage = mpimg.imread(os.path.join(path,fileNames[i]))
        i+=1
        plt.axis('off')
        plt.imshow(firstImage)
    return i

#load dirs from path
def load_folders(path, classCount):
    foldersArray = []
    #folders = os.listdir(path)
    i=1
    fig = plt.figure()
    folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for file in folders[:classCount]:
        i =load_pictures(path + file,fig,i)

        foldersArray.append(path + file)
    plt.show()
    return  foldersArray


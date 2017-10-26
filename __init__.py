import  LoadFiles, DataCompresion, DataDivision, VerifyQuality, FinalShowImg,Hashing, KMeans, DBScan, SOM
from six.moves import cPickle as pickle

# path = 'C:/Users/andrej.duben/PycharmProjects/SUNS1/notMNIST_large/'
path = 'C:/Users/andrej.duben/PycharmProjects/SUNS1/notMNIST_small/'

classCount = 10
#validation, test, train = []


files = LoadFiles.load_folders(path, classCount)
pickles = DataCompresion.toPickle(files)
VerifyQuality.verify(pickles)
dividedFile = DataDivision.splitToSets(pickles)
pathToFinalPickle = DataDivision.toOneFile(path,dividedFile)
#FinalShowImg.FinalShowValid(pathToFinalPickle)
#FinalShowImg.FinalShowTest(pathToFinalPickle)
#FinalShowImg.FinalShowValid(pathToFinalPickle)

with open(pathToFinalPickle, 'rb') as f:
    unpickled = pickle.load(f)

    datasets = unpickled.values()


# Hashing.hashSmall(unpickled.get('validation_DS'),unpickled.get('test_DS'),unpickled.get('train_DS'),unpickled.get('validation_CD'))
KMeans.KmeansAlgorithm(unpickled.get('train_DS'))
#DBScan.DBScanAlghoritm(unpickled.get('validation_DS'))
# SOM.SOMAlghoritm(unpickled.get('train_DS'))
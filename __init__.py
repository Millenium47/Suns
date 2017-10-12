import  LoadFiles, DataCompresion, DataDivision, VerifyQuality, FinalShowImg

path = 'C:/Users/andrej.duben/PycharmProjects/SUNS1/notMNIST_small/'
classCount = 10

files = LoadFiles.load_folders(path, classCount)
pickles = DataCompresion.toPickle(files)
VerifyQuality.verify(pickles)
dividedFile = DataDivision.splitToSets(pickles)
pathToFinalPickle = DataDivision.toOneFile(path,dividedFile)
FinalShowImg.FinalShowValid(pathToFinalPickle)
FinalShowImg.FinalShowTest(pathToFinalPickle)
FinalShowImg.FinalShowValid(pathToFinalPickle)
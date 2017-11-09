import LoadFiles, DataCompresion, DataDivision, VerifyQuality, FinalShowImg, Hashing, KMeans, DBScan, SOM, SVM, MLP

from six.moves import cPickle as pickle
import matplotlib.pyplot as plt

path = 'C:/Users/andrej.duben/PycharmProjects/SUNS/notMNIST_large/'
# path = 'C:/Users/andrej.duben/PycharmProjects/SUNS/notMNIST_small/'

classCount = 10
# validation, test, train = []


# files = LoadFiles.load_folders(path, classCount)
# pickles = DataCompresion.toPickle(files)
# VerifyQuality.verify(pickles)
# dividedFile = DataDivision.splitToSets(pickles)
# pathToFinalPickle = DataDivision.toOneFile(path, dividedFile)
# FinalShowImg.FinalShowTrain(pathToFinalPickle)
# FinalShowImg.FinalShowTest(pathToFinalPickle)
# FinalShowImg.FinalShowValid(pathToFinalPickle)

# -------------------- CVICENIE 2 -----------------------

with open(path + 'Final.pickle', 'rb') as f:
    unpickled = pickle.load(f)

filtered_train_data = Hashing.hashSmall(unpickled.get('validation_DS'), unpickled.get('test_DS'),
                                        unpickled.get('train_DS'), unpickled.get('train_CD'))
# KMeans.KmeansAlgorithm(unpickled.get('train_DS'))
# DBScan.DBScanAlghoritm(unpickled.get('validation_DS'))
# SOM.SOMAlghoritm(unpickled.get('train_DS'))

# -------------------- CVICENIE 3 -----------------------


# mlp_class = MLP.mlp_train(unpickled.get('train_DS'), unpickled.get('train_CD'), 1000, (500,))
# MLP.mlp_test(unpickled.get('test_DS'), unpickled.get('test_CD'), 2800, mlp_class)

amounts_of_training_data = [50, 100, 1000, 5000, 20000]
legend = []

accuracies_hidden_1 = []
accuracies_hidden_n = []
# # unfiltered mlp
# for data_size in amounts_of_training_data:
#     mlp = MLP.mlp_train(unpickled.get('train_DS'), unpickled.get('train_CD'), data_size, (500,))
#     acc = MLP.mlp_test(unpickled.get('test_DS'), unpickled.get('test_CD'), 10000, mlp)
#     accuracies_hidden_1.append(acc)
#
# plt.plot(amounts_of_training_data, accuracies_hidden_1, marker='o')
# legend.append('Unfiltered: 1 Hidden Layer')
# accuracies_hidden_1, accuracies_hidden_n = [], []
# # unfiltered N mpl
# for data_size in amounts_of_training_data:
#     mlp = MLP.mlp_train(unpickled.get('train_DS'), unpickled.get('train_CD'), data_size, (500, 500))
#     acc = MLP.mlp_test(unpickled.get('test_DS'), unpickled.get('test_CD'), 10000, mlp)
#     accuracies_hidden_1.append(acc)
#
# plt.plot(amounts_of_training_data, accuracies_hidden_1, marker='o')
# legend.append('Unfiltered: n Hidden Layer')
#
# accuracies_hidden_1, accuracies_hidden_n = [], []
# # fitlered mlp
# for data_size in amounts_of_training_data:
#     mlp = MLP.mlp_train(filtered_train_data[0], filtered_train_data[1], data_size, (500,))
#     acc = MLP.mlp_test(unpickled.get('test_DS'), unpickled.get('test_CD'), 10000, mlp)
#     accuracies_hidden_1.append(acc)
#
# plt.plot(amounts_of_training_data, accuracies_hidden_1, marker='o')
# legend.append('Filtered: 1 Hidden Layer')
# accuracies_hidden_1, accuracies_hidden_n = [], []
# # fitlered N mlp
# for data_size in amounts_of_training_data:
#     mlp = MLP.mlp_train(filtered_train_data[0], filtered_train_data[1], data_size, (500, 500))
#     acc = MLP.mlp_test(unpickled.get('test_DS'), unpickled.get('test_CD'), 10000, mlp)
#     accuracies_hidden_1.append(acc)
#
# plt.plot(amounts_of_training_data, accuracies_hidden_1, marker='o')
# legend.append('Filtered: N Hidden Layer')
# accuracies_hidden_1, accuracies_hidden_n = [], []
# # draw
# plt.suptitle('MLP results')
# plt.legend(legend, loc='lower right')
# plt.show()

# -------------------- SVM  -----------------------
# svm_class = SVM.svm_linear_train(unpickled.get('train_DS'), unpickled.get('train_CD'), 1000)
# SVM.svm_test(unpickled.get('test_DS'), unpickled.get('test_CD'), 2800, svm_class)
legend = []
# SVM linear unfiltered
for data_size in amounts_of_training_data:
    svm = SVM.svm_linear_train(unpickled.get('train_DS'), unpickled.get('train_CD'), data_size)
    acc = SVM.svm_test(unpickled.get('test_DS'), unpickled.get('test_CD'), 10000, svm)
    accuracies_hidden_1.append(acc)

plt.plot(amounts_of_training_data, accuracies_hidden_1, marker='o')
legend.append('Unfiltered: 1 Hidden Layer')
accuracies_hidden_1, accuracies_hidden_n = [], []

# SVM linear filtered
for data_size in amounts_of_training_data:
    svm = SVM.svm_linear_train(filtered_train_data[0], filtered_train_data[1], data_size)
    acc = SVM.svm_test(unpickled.get('test_DS'), unpickled.get('test_CD'), 10000, svm)
    accuracies_hidden_1.append(acc)

plt.plot(amounts_of_training_data, accuracies_hidden_1, marker='o')
legend.append('Unfiltered: 1 Hidden Layer')
accuracies_hidden_1, accuracies_hidden_n = [], []

# SVM poly unfiltered
for data_size in amounts_of_training_data:
    svm = SVM.svm_poly_train(unpickled.get('train_DS'), unpickled.get('train_CD'), data_size)
    acc = SVM.svm_test(unpickled.get('test_DS'), unpickled.get('test_CD'), 10000, svm)
    accuracies_hidden_1.append(acc)

plt.plot(amounts_of_training_data, accuracies_hidden_1, marker='o')
legend.append('Unfiltered: 1 Hidden Layer')
accuracies_hidden_1, accuracies_hidden_n = [], []

# SVM poly filtered
for data_size in amounts_of_training_data:
    svm = SVM.svm_poly_train(filtered_train_data[0], filtered_train_data[1], data_size)
    acc = SVM.svm_test(unpickled.get('test_DS'), unpickled.get('test_CD'), 10000, svm)
    accuracies_hidden_1.append(acc)

plt.plot(amounts_of_training_data, accuracies_hidden_1, marker='o')
legend.append('Unfiltered: 1 Hidden Layer')
accuracies_hidden_1, accuracies_hidden_n = [], []

# draw
plt.suptitle('SVM results')
plt.legend(legend, loc='lower right')
plt.show()
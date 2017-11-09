import MLP, SVM
import matplotlib.pyplot as plt


def train_and_classify(train, trainL, ftrain, ftrainL, test, testL):
    dataCycles = [50, 100, 1000, 5000, 20000]
    predictions = []

    # for dataCycle in dataCycles:
    #     mlp = MLP.mlp_train(train, trainL, dataCycle, (100, 100))
    #     prediction = MLP.mlp_test(test, testL, 10000, mlp)
    #     predictions.append(prediction)
    #
    #     plt.plot(dataCycle, predictions, marker='*')
    #
    amounts_of_training_data = [50, 100, 1000, 5000]
    legend = []

    accuracies_hidden_1 = []
    accuracies_hidden_n = []

    for data_size in amounts_of_training_data:
        mlp = MLP.mlp_train(train,trainL,data_size, (5,2))
        acc = MLP.mlp_test(test,testL, 1000, mlp)
        accuracies_hidden_1.append(acc)

    plt.plot(amounts_of_training_data, accuracies_hidden_1, marker='o')
    legend.append('Unfiltered: 1 Hidden Layer')
    #
    # for data_size in amounts_of_training_data:
    #     mlp = MLP.mlp_train(train, trainL, data_size, (100,100))
    #     acc = MLP.mlp_test(test, testL, 1000, mlp)
    #     accuracies_hidden_n.append(acc)
    #
    # plt.plot(amounts_of_training_data, accuracies_hidden_n, marker='o')
    # legend.append('Unfiltered: 2 Hidden Layer')
    #
    #
    plt.suptitle('MLP results')
    plt.legend(legend, loc='lower right')
    plt.show()

    # #SVM
    # legend = []
    #
    # accuracies_SVM = []
    # accuracies_hidden_n = []
    #
    # for data_size in amounts_of_training_data:
    #     svm = SVM.svm_train(train, trainL, data_size)
    #     acc = SVM.svm_test(test, testL, 1000, svm)
    #     accuracies_SVM.append(acc)
    #
    # plt.plot(amounts_of_training_data, accuracies_SVM, marker='o')
    # legend.append('Unfiltered: SVM')
    # #
    # # for data_size in amounts_of_training_data:
    # #     mlp = MLP.mlp_train(train, trainL, data_size, (100,100))
    # #     acc = MLP.mlp_test(test, testL, 1000, mlp)
    # #     accuracies_hidden_n.append(acc)
    # #
    # # plt.plot(amounts_of_training_data, accuracies_hidden_n, marker='o')
    # # legend.append('Unfiltered: 2 Hidden Layer')
    # #
    # #
    # plt.suptitle('SVM results')
    # plt.legend(legend, loc='lower right')
    # plt.show()

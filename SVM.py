import os
import matplotlib.pyplot as plt
from sklearn import svm


def svm_linear_train(train_dataset, training_labels, data_size):
    X = train_dataset[:data_size, :, :]
    if X.shape[1] == 28:
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = training_labels[:data_size]

    svm_classificator = svm.SVC(kernel='linear', C=1.0)

    svm_classificator.fit(X, y)
    print('---------------------------------------------------')
    print('SVM trained for ' + str(data_size))
    return svm_classificator

def svm_poly_train(train_dataset, training_labels, data_size):
    X = train_dataset[:data_size, :, :]
    if X.shape[1] == 28:
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = training_labels[:data_size]

    svm_classificator = svm.SVC(kernel='poly', degree=3, C=1.0)

    svm_classificator.fit(X, y)
    print('---------------------------------------------------')
    print('SVM trained for ' + str(data_size))
    return svm_classificator


def svm_test(test_dataset, test_labels, testdata_count, svm_classificator):
    data = test_dataset[:testdata_count,:,:]
    if data.shape[1] == 28:
        data = data.reshape(data.shape[0], data.shape[1]*data.shape[2])
    labels = test_labels[:testdata_count]
    prediction = svm_classificator.predict(data)

    true_predictions = 0
    false_predictions = 0

    for i, result in enumerate(prediction):
        if result == labels[i]:
            true_predictions += 1
        else:
            false_predictions +=1

        prediction_rate = round((true_predictions / (false_predictions + true_predictions)) * 100, 3)

    print('SVMClassifier predicition accuracy: '+str(prediction_rate)+' for ' + str(testdata_count))
    print('---------------------------------------------------')

    return prediction_rate

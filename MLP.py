import os
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def mlp_train(train_dataset, training_labels, data_size, layers):
    X = train_dataset[:data_size, :, :]
    # if X.shape[1] == 28:
    X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    y = training_labels[:data_size]
#     # ===============
#     fig = plt.figure()
#     plt.title('Testing data')
#     plt.axis('off')
#     i = 1
#     j = 15
#
#     for image in train_dataset[15:18]:
#         a = fig.add_subplot(1, 3, i)
#         a.title.set_text(training_labels[j])
#         i += 1
#         j += 1
#
#         plt.axis('off')
#         plt.imshow(image)
#
#
#     plt.show()
# # ===================

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
# mlp_classificator = MLPClassifier(hidden_layer_sizes=layers, solver ='lbfgs', alpha=1e-5, random_state=1)
    mlp_classificator = MLPClassifier(hidden_layer_sizes=layers, solver='sgd', batch_size='auto',
                                  learning_rate='constant', learning_rate_init=0.09, max_iter=500, shuffle=True,
                                  random_state=1, momentum=0.9, early_stopping=True, nesterovs_momentum=True)

    mlp_classificator.fit(X, y)
    print('---------------------------------------------------')
    print('MLP trained for ' + str(data_size))
    return mlp_classificator


def mlp_test(test_dataset, test_labels, testdata_count, mlp_classificator):
    data = test_dataset[:testdata_count, :, :]
    data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
    labels = test_labels[:testdata_count]
    prediction = mlp_classificator.predict(data)

    true_predictions = 0
    false_predictions = 0

    for i, result in enumerate(prediction):
        if result == labels[i]:
            true_predictions += 1
        else:
            false_predictions += 1

    prediction_rate = round((true_predictions / testdata_count) * 100, 3)

    print('MLPClassifier predicition accuracy: ' + str(prediction_rate) + ' for ' + str(testdata_count))
    print('---------------------------------------------------')

    return prediction_rate

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt

from adaline import Adaline


def training(first_label, second_label, first_letter, second_letter):
    first_label = [1] * len(first_label)
    second_label = [-1] * len(second_label)

    # combine labels and letters for "bet" and "mem" categories
    labels_combined = np.concatenate((first_label, second_label))
    letters_combined = np.concatenate((first_letter, second_letter))

    # shuffle the data
    combined_data = np.random.permutation(len(labels_combined))
    labels_combined = labels_combined[combined_data]
    letters_combined = letters_combined[combined_data]
    accuracies = []
    # split the data into 5 equal parts
    chunk = StratifiedKFold(n_splits=5)
    for train_index, test_index in chunk.split(letters_combined, labels_combined):
        X_train, X_test = letters_combined[train_index], letters_combined[test_index]
        y_train, y_test = labels_combined[train_index], labels_combined[test_index]
        adaline = Adaline()
        adaline.fit(X_train, y_train)
        predict = adaline.predict(X_test)
        accuracy = np.mean(predict == y_test)
        accuracies.append(accuracy)
        cm = confusion_matrix(y_test, predict)
        # define class labels
        class_labels = ['first_label', 'second_label']
        # create heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="pink", xticklabels=class_labels, yticklabels=class_labels)
        # add labels and title
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        # display the plot
        plt.show()

        # print("Accuracy:", accuracy)
        # print("Predict:", cm)

    print(f'Average accuracy: {np.mean(accuracies):.2f}')
    print(f'Standard deviation: {np.std(accuracies):.2f}')

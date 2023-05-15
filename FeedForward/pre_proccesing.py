import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from feedForward import create_feedforward_nn

def training(first_label, second_label, first_letter, second_letter):
    first_label = np.array(first_label)
    second_label = np.array(second_label)

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

        # convert labels to categorical format
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # create the feed-forward neural network model
        input_dim = X_train.shape[1]
        num_classes = y_train.shape[1]
        model = create_feedforward_nn(input_dim, num_classes)

        # fit the model to the training data
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        # evaluate the model on the test data
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(accuracy)

        # generate predictions on the test data
        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)

        # create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
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

    print(f'Average accuracy: {np.mean(accuracies) * 100:.2f}')
    print(f'Standard deviation: {np.std(accuracies):.2f}')

from sklearn.metrics import confusion_matrix

from adaline import Adaline
import os
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':

    directory = '/Users/ronipick/PycharmProjects/nuiro1/data'

    # creating the dataset
    label_mem = []
    letter_mem = []
    label_bet = []
    letter_bet = []
    label_lamed = []
    letter_lamed = []

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Consider only text files
            file_path = os.path.join(directory, filename)

            with open(file_path, 'r') as file:
                lines = file.readlines()
                # fitting the data
                for line in lines:
                    line = line.replace(' ', '')
                    line = line.replace('(', '')
                    line = line.replace(')', '')
                    line = line.replace('\n', '')

                    line = line.split(',')

                    # checking if the first element is in our range, is fo - append the letter's label to their category
                    if line[0] in ['1', '2', '3'] and len(line) == 101:
                        if line[0] == '1':
                            label_bet.append(1)
                            letter_bet.append(line[1:])
                        elif line[0] == '2':
                            label_lamed.append(2)
                            letter_lamed.append(line[1:])
                        elif line[0] == '3':
                            label_mem.append(3)
                            letter_mem.append(line[1:])

    # Convert the letters list to a NumPy array
    letter_mem = np.array(letter_mem, dtype=int)
    letter_bet = np.array(letter_bet, dtype=int)
    letter_lamed = np.array(letter_lamed, dtype=int)

    label_mem = [1] * len(letter_mem)
    label_lamed = [-1] * len(letter_lamed)

    # Combine labels and letters for "bet" and "mem" categories
    labels_combined = np.concatenate((label_lamed, label_mem))
    letters_combined = np.concatenate((letter_lamed, letter_mem))

    # Shuffle the data
    combined_data = np.random.permutation(len(labels_combined))
    labels_combined = labels_combined[combined_data]
    letters_combined = letters_combined[combined_data]

    # split the data into 5 equal parts
    chunk = StratifiedKFold(n_splits=5)
    for train_index, test_index in chunk.split(letters_combined, labels_combined):
        X_train, X_test = letters_combined[train_index], letters_combined[test_index]
        y_train, y_test = labels_combined[train_index], labels_combined[test_index]
        adaline = Adaline()
        adaline.fit(X_train, y_train)
        predict = adaline.predict(X_test)
        accuracy = np.mean(predict == y_test)
        cm = confusion_matrix(y_test, predict)
        print("Accuracy:", accuracy)
        print("Predict:", cm)

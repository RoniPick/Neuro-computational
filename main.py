from adaline import Adaline
import os
import random
import numpy as np

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
                    if line[0] in ['1', '2', '3']:
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
    label_bet = [-1] * len(letter_bet)

    # Combine labels and letters for "bet" and "mem" categories
    labels_combined = np.concatenate((label_bet, label_mem))
    letters_combined = np.concatenate((letter_bet, letter_mem))

    print("labels_combined: ", labels_combined)

    # print("labels_combined: ", labels_combined)
    # print("letters_combined: ", letters_combined)

    # Shuffle the data
    combined_data = np.column_stack((labels_combined, letters_combined))
    np.random.shuffle(combined_data)
    # print("combined_data: ", combined_data)

    # Split the shuffled data back into labels and letters
    shuffled_labels = combined_data[:, 0]
    shuffled_letters = combined_data[:, 1:]
    # print("shuffled_labels: ", shuffled_labels)
    # print("shuffled_letters: ", shuffled_letters)

    # Split data into training and test sets
    train_ratio = 0.8  # Percentage of data for training
    split_index = int(train_ratio * len(shuffled_labels))

    X_train = shuffled_letters[:split_index]
    y_train = shuffled_labels[:split_index]

    X_test = shuffled_letters[split_index:]
    y_test = shuffled_labels[split_index:]

    # X_train = shuffled_letters  # the training data
    # y_train = shuffled_labels  # the training labels

    # X_test = ...  # Your test data
    # y_test = ...  # Your test labels

    # step 2 - create the adaline classifier
    adaline = Adaline()
    adaline.fit(X_train, y_train)

    train_accuracy = adaline.accuracy(X_train, y_train)
    test_accuracy = adaline.accuracy(X_test, y_test)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

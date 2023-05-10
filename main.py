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

                    line = line.split(',')

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

                        # else:
                        #     print("The file does not meet the specified conditions.")

    np.array(letter_mem)
            # np.array(label_mem)
    np.array(letter_bet)
            # np.array(label_bet)
    np.array(letter_lamed)
            # np.array(label_lamed)

    # creating the dataset


    # createDataSet(file)

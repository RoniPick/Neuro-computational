import os
import random
import numpy as np


# creating the dataset
def createDataSet(file, label_bet, label_lamed, label_mem, letter_bet, letter_lamed, letter_mem):

    with open(file, 'r') as file:
        lines = file.readlines()
        # fitting the data to the algorithm
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

    return label_bet, label_lamed, label_mem, letter_bet, letter_lamed, letter_mem

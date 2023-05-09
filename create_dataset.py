import random
import numpy as np


# creating the dataset
def createDataSet(file, labels, letters):
    with open(file) as f:
        content = f.readlines()
        # creating the dataset
        for line in content:
            if line[1] in ['1', '2', '3']:
                line = line.strip()
                modified_text = '(' + line[4:]  # remove characters at positions 1-3 - the labels
                labels.append(line[1])  # 1 = bet , 2 = lamed , 3 = mem
                letters.append(modified_text)
            else:
                print("The file does not meet the specified conditions.")



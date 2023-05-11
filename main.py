
from create_dataset import createDataSet
from pre_proccesing import training

if __name__ == '__main__':

    directory = '/Users/ronipick/PycharmProjects/nuiro1/data'

    # creating the dataset
    label_mem = []
    letter_mem = []
    label_bet = []
    letter_bet = []
    label_lamed = []
    letter_lamed = []

    label_bet, label_lamed, label_mem, letter_bet, letter_lamed, letter_mem = createDataSet(directory, label_bet,
                                                                                            label_lamed, label_mem,
                                                                                            letter_bet, letter_lamed,
                                                                                            letter_mem)

    print("letter mem vs letter lamed")
    training(label_mem, label_lamed, letter_mem, letter_lamed)
    print("\nletter mem vs letter bet")
    training(label_mem, label_bet, letter_mem, letter_bet)
    print("\nletter lamed vs letter bet")
    training(label_lamed, label_bet, letter_lamed, letter_bet)

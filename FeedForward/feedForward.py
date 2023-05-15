from keras.models import Sequential
from keras.layers import Dense


def create_feedforward_nn(input_dim, num_classes):
    model = Sequential()  # create a sequential model

    # add a dense layer with 128 units and 'relu' activation function, input dimension is set to 'input_dim'
    model.add(Dense(128, activation='relu', input_dim=input_dim))

    # add a dense layer with 64 units and 'relu' activation function
    model.add(Dense(64, activation='relu'))

    # add a dense layer with 'num_classes' units and 'softmax' activation function, 'num_classes' represents the
    # number of output classes
    model.add(Dense(num_classes, activation='softmax'))

    # compile the model with 'adam' optimizer, 'categorical_crossentropy' loss function, and use 'accuracy' as the
    # metric for evaluation
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

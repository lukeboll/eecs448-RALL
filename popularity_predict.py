import random
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout

random.seed(6161)

# Parameters
DATA_LEN = 1600
DATA_WIDTH = 942
STEP = 100
DROPOUT_RATE = 0.25
RESOURCE_BARRIER = 200

# Functions
def make_data(f_X, f_y):
    X = []
    y = []
    with open(f_X, 'r', encoding='utf-8') as fx, open(f_y, 'r', encoding='utf-8') as fy:
        for s in fx.readlines()[1:]:
            if s[0] == 'A':
                X.append(list(map(float, sav)))
                sav = []
            else:
                sav.append(float(s))
        for s in fy:
            y.append(float(s))
    return X, y

def make_popularity_model():
    model = Sequential()
    data_len = DATA_LEN
    cnt = STEP
    while cnt < data_len and data_len - cnt >= RESOURCE_BARRIER:
        model.add(Conv1D(data_len-cnt, kernel_size=cnt, activation='relu'))
        data_len = int(data_len*(1-DROPOUT_RATE))
        cnt += STEP
    model.add(Conv1D(1, kernel_size=cnt-STEP, activation='relu'))
    model.build(input_shape=(DATA_WIDTH, DATA_LEN, 1))
    return model

def train_and_test(filename, model):
    X, y = make_data('data/audios_record.txt', filename)
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)

    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY)
    score = model.evaluate(testX, testY, verbose=0)
    print("Train accuracy:", score[0])
    print("Test accuracy:", score[1])

# Main
popularity_model = make_popularity_model()
train_and_test('data/popularity.txt', popularity_model)

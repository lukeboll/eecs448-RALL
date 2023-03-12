import random

import keras
import numpy as np
from keras.layers import (Conv1D, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling1D, MaxPooling2D)
from keras.models import Sequential
from sklearn.model_selection import train_test_split

random.seed(6161)


def make_data(f_X, f_y):
    X = []
    y = []
    with open(f_X, 'r', encoding='utf-8') as fx:
        cnt_group = 0
        sav = []
        flg = True
        for s in fx:
            if flg:
                flg = False
                continue  # skip the first line
            if s[0] == 'A':
                X.append(sav.copy())
                sav = []
                cnt_group += 1
            else:
                sav.append(float(s))
    with open(f_y, 'r', encoding='utf-8') as fy:
        for s in fy:
            y.append(float(s))
    return X, y


DATA_LEN = 1600
data_size = 942
step = 100
dropout_rate = 0.25
resource_barrier = 200


def make_valence_model():
    valence_model = Sequential()
    data_len = DATA_LEN
    cnt = step
    while cnt < data_len and data_len - cnt >= resource_barrier:
        valence_model.add(
            Conv1D(data_len-cnt, kernel_size=cnt, activation='relu'))
        data_len = int(data_len*(1-dropout_rate))
        cnt += step
    valence_model.add(Conv1D(1, kernel_size=cnt-step, activation='relu'))
    valence_model.build(input_shape=(data_size, DATA_LEN, 1))
    return valence_model


def make_arousal_model():
    arousal_model = Sequential()
    data_len, cnt = DATA_LEN, step

    while cnt < data_len and data_len - cnt >= resource_barrier:
        arousal_model.add(
            Conv1D(data_len-cnt, kernel_size=cnt, activation='relu'))
        data_len = int(data_len*(1-dropout_rate))
        cnt += step
    arousal_model.add(Conv1D(1, kernel_size=cnt-step, activation='relu'))
    arousal_model.build(input_shape=(data_size, DATA_LEN, 1))
    return arousal_model


def train_and_test(filename, model):
    X, y = make_data('data/audios_record.txt', filename)
    testing_size = int(len(X)*0.2)
    training_size = len(X) - testing_size
    trainX, testX, trainY, testY = train_test_split(
        X, y, test_size=testing_size, train_size=training_size)

    model.compile(loss=keras.losses.MeanAbsoluteError(),
                  optimizer='adam', metrics=['accuracy'])
    model.fit(trainX, trainY)
    score = model.evaluate(testX, testY, verbose=0)
    print("Train accuracy: "+str(score[0]))
    print("Test accuracy: "+str(score[1]))


valence_model = make_valence_model()
arousal_model = make_arousal_model()

train_and_test('data/arousal.txt', arousal_model)
train_and_test('data/valence.txt', valence_model)

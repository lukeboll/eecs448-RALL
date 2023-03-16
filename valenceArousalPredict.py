import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,Conv1D, MaxPooling2D, MaxPooling1D
import numpy as np
from sklearn.model_selection import train_test_split
import random
random.seed(6161)
genre_tags = ['ROCK', 'JAZZ', 'POP', 'ELEC', 'WORLD','COUNTRY']
DATA_LEN = 996
FEATURE_LEN = 26
test_ratio = 0.2
data_size = 942
train_data_size = data_size - int(data_size * test_ratio)
dropout_rate = 0.25
resource_barrier = 200

def makeRawData(f_X, f_y,lim=2010):
    X = np.zeros((data_size, DATA_LEN, FEATURE_LEN))
    y = np.zeros(data_size)
    with open(f_X,'r',encoding='utf-8') as fx:
        cnt_group = 0
        cnt_dat = 0
        sav = np.zeros((DATA_LEN, FEATURE_LEN))
        flg = True
        for s in fx:
            if cnt_group >= lim:
                break
            if flg:
                flg = False
                continue # skip the first line
            if s[0]=='A':
                X[cnt_group,:,:] = sav
                sav = np.zeros((DATA_LEN, FEATURE_LEN))
                cnt_group += 1
                cnt_dat = 0
            else:
                s=s.split(' ')
                cur_sav = np.zeros(FEATURE_LEN)
                counter = 0
                for w in s:
                    if len(w) <=1:
                        continue
                    cur_sav[counter] = float(w)
                    counter += 1
                sav[cnt_dat,:] = cur_sav
    with open(f_y, 'r', encoding='utf-8') as fy:
        cnt_group = 0
        for s in fy:
            y[cnt_group] = float(s)
            cnt_group += 1
    return X,y


def makeValenceModel(layer_cnt = 3):
    input_layer = keras.Input(shape=(DATA_LEN,FEATURE_LEN))
    step_size, k_size = 300, 250
    cnt = 1
    layer = Conv1D(DATA_LEN - cnt*step_size, kernel_size = k_size)(input_layer)
    while cnt<layer_cnt:
        cnt += 1
        layer = Conv1D(DATA_LEN - cnt*step_size,kernel_size = k_size, activation='selu')(layer)
        layer = Dropout(0.2)(layer)
    output_layer = Dense(1, activation='linear')(layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())
    config = model.get_config() # Returns pretty much every information about your model
    print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
    [print(i.shape, i.dtype) for i in model.inputs]
    [print(o.shape, o.dtype) for o in model.outputs]
    [print(l.name, l.input_shape, l.dtype) for l in model.layers]
    return model

def makeArousalModel(layer_cnt = 3):
    input_layer = keras.Input(shape=(DATA_LEN,FEATURE_LEN))
    step_size, k_size = 300, 250
    cnt = 1
    layer = Conv1D(DATA_LEN - cnt*step_size, kernel_size = k_size)(input_layer)
    while cnt<layer_cnt:
        cnt += 1
        layer = Conv1D(DATA_LEN - cnt*step_size,kernel_size = k_size, activation='selu')(layer)
        layer = Dropout(0.2)(layer)
    output_layer = Dense(1, activation='linear')(layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())
    config = model.get_config() # Returns pretty much every information about your model
    print(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
    [print(i.shape, i.dtype) for i in model.inputs]
    [print(o.shape, o.dtype) for o in model.outputs]
    [print(l.name, l.input_shape, l.dtype) for l in model.layers]
    return model

def trainAndTest(filename, model, name):
    X,y = makeRawData('audios_record.txt',filename)
    print(X.shape)
    print(y.shape)
    testing_size = int(len(X)*test_ratio)
    training_size = len(X) - testing_size
    trainX, testX, trainY, testY = train_test_split(X,y,test_size=testing_size,train_size=training_size)
    print("splitting done")
    model.compile(loss=keras.losses.MeanAbsoluteError(),optimizer='adam',metrics=['mean_absolute_error'])
    model.fit(trainX, trainY)
    print("fitting done")
    model.save(name)
    score = model.evaluate(testX, testY, verbose=0)
    print(model.metrics_names)
    print(score)

valence_model = None
arousal_model = makeArousalModel()
valence_model = makeValenceModel()
trainAndTest('arousal.txt',arousal_model, "arousal_model")
trainAndTest('valence.txt',valence_model, "valence_model")
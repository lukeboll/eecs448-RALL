#If no data accompanied, this file should be used along with the following files/folders:
#   genre_model
#   valence_model
#   arousal_linear_model.pkl
#   valence_linear_model.pkl

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,Conv1D, LSTM,Bidirectional
import numpy as np
from python_speech_features import mfcc
import librosa
from sklearn.model_selection import train_test_split
import random
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import os
import joblib

random.seed(6161)
genre_tags = ['ROCK', 'JAZZ', 'POP', 'ELEC', 'WORLD','COUNTRY']
DATA_LEN = 996
FEATURE_LEN = 26
test_ratio = 0.2
data_size = 942
train_data_size = data_size - int(data_size * test_ratio)
dropout_rate = 0.25
resource_barrier = 200

def genMFCCFeatures(signal, samplerate=16000):
    mfcc_features = mfcc(signal,
                         samplerate,
                         winlen = 0.05,
                         winstep = 0.01,
                         numcep = 50,
                         nfft = 1024)
    mfcc_features = preprocessing.scale(mfcc_features)
    return mfcc_features

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

def RNNGenreModel():
    input_layer = keras.Input(shape=(DATA_LEN,FEATURE_LEN))
    layer = Bidirectional(LSTM(units=64))(input_layer)
    layer = Dense(units=64)(layer)
    layer = Dense(units=32)(layer)
    layer = Dropout(rate=0.2)(layer)
    output_layer = Dense(1, activation='linear')(layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())
    model.save('genre_model')
    return model

def trainAndTestGenre(input_file, filename, model, name):
    X,y = makeRawData(input_file,filename)
    testing_size = int(len(X)*test_ratio)
    training_size = len(X) - testing_size
    trainX, testX, trainY, testY = train_test_split(X,y,test_size=testing_size,train_size=training_size)
    print("splitting done")
    model.compile(loss=keras.losses.MeanSquaredError(),optimizer='adam',metrics=['accuracy'])
    model.fit(trainX, trainY)
    print("fitting done")
    model.save(name)
    score = model.evaluate(testX, testY, verbose=0)
    print(model.metrics_names)
    print(score)

def trainAndTest(input_file, filename, model, name):
    X,y = makeRawData(input_file,filename)
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

#def makeRegressionModel(am, vm, gm):
    
def load_data(input_file):
  #reading the file
  aud, sample_rate = librosa.load(input_file)
  #make it 1-D
  if len(aud.shape) > 1: 
      aud = aud[:,0] + aud[:,1]
  aud= librosa.resample(aud,orig_sr=sample_rate,target_sr=16000)
  sample_rate = 16000
  return aud,sample_rate

def splitAudio(filepath,segment = 15, step = 3):
    newAudio,rate = load_data(filepath)
    audio_len = len(newAudio)
    ret = [], time_span = []
    for i in range(step*rate,audio_len - step*rate,step*rate):
        l = i - segment*rate
        r = i
        sav_aud = newAudio[l:r]
        ret.append(sav_aud)
        time_span.append([l,r])
    return ret, time_span

def multimodelTraining(song_path,valence_model, arousal_model, genre_model,regression_model):
    segments,time_span = splitAudio(song_path)
    X_arousal = []
    X_valence = []
    for i in range(len(segments)):
        seg, t = segments[i], time_span[i]
        feature = genMFCCFeatures(seg)
        v = valence_model(feature)
        a = arousal_model(feature)
        g = genre_model(feature)
        tempo, beats = librosa.beat.beat_track(y=seg, sr=16000)
        X_arousal.append([a,beats,g])
        X_valence.append([v,beats,g])
    y_arousal = []
    y_valence = []
    for i in range(len(segments)):
        y_arousal.append(regression_model(X_arousal[i]))
        y_valence.append(regression_model(X_valence[i]))

    return y_arousal, y_valence

def integrateModel(signal,sr,V,A,G):
    signal = signal[15*sr:25*sr]
    signal = np.asarray(signal,dtype=float)
    tempo, beats = librosa.beat.beat_track(y=signal, sr=16000)
    signal = np.asarray([genMFCCFeatures(signal)],dtype=float)
    l1 = [np.mean(A.predict(signal)[0,:,0]),int(G.predict(signal)+0.5),tempo]
    l2 = [np.mean(V.predict(signal)[0,:,0]),int(G.predict(signal)+0.5),tempo]

    return l1, l2

def makeRegressionModel(V,A,G,data_len = 200):
    V.compile(loss=keras.losses.MeanAbsoluteError(),optimizer='adam',metrics=['mean_absolute_error'])
    A.compile(loss=keras.losses.MeanAbsoluteError(),optimizer='adam',metrics=['mean_absolute_error'])
    G.compile(loss=keras.losses.MeanSquaredError(),optimizer='adam',metrics=['accuracy'])
    all_valence, all_arousal, val_features, aro_features= [],[],[],[]
    X,valence_vals = makeRawData('audios_record.txt','valence.txt')
    X,arousal_vals = makeRawData('audios_record.txt','arousal.txt')
    ids = []
    with open('id_record.txt','r',encoding='utf-8') as f_in:
        data = f_in.read()
        data = data.split('\n')
        for dat in data:
            if(dat != ''):
                ids.append(int(dat))
    cnt = 0
    for i in range(data_len):
        if i%100 == 0:
            print('processing '+str(i))
        if(i != ids[cnt]):
            continue
        filename = str(ids[cnt])+".wav"
        if os. path. isfile(filename):
            signal,sr = load_data(filename)
            l1,l2 = integrateModel(signal,sr, V,A,G)
            all_valence.append(valence_vals[cnt])
            all_arousal.append(arousal_vals[cnt])
            val_features.append(l1)
            aro_features.append(l2)
            cnt += 3

    arousal_linear_model = LinearRegression()
    valence_linear_model = LinearRegression()
    arousal_linear_model.fit(aro_features, all_arousal)
    joblib.dump(arousal_linear_model, "arousal_linear_model.pkl") 
    valence_linear_model.fit(val_features, all_valence)
    joblib.dump(valence_linear_model, "valence_linear_model.pkl") 
    return valence_linear_model, arousal_linear_model


def getTrends(songfile, V,A,G,vlin,alin, step = 1):
    song,rate = load_data(songfile)
    cnt,cur_segment,segment_len = 0, 10 * rate, 10 * rate
    valence_trends, arousal_trends = [],[]
    while(cur_segment+segment_len <= len(song)):
        l,r = cur_segment, cur_segment + segment_len
        cur_song = song[l:r]
        tempo, beats = librosa.beat.beat_track(y=np.asarray(cur_song), sr=16000)
        features = np.asarray([genMFCCFeatures(cur_song)],dtype=float)
        valence_val = vlin.predict([[np.mean(V.predict(features)[0,:,0]),int(G.predict(features)+0.5),tempo]])
        arousal_val = alin.predict([[np.mean(A.predict(features)[0,:,0]),int(G.predict(features)+0.5),tempo]])
        valence_trends.append(valence_val[0])
        arousal_trends.append(arousal_val[0])
        cur_segment += step*rate
    return valence_trends, arousal_trends
#valence_model = None
#arousal_model = makeArousalModel()
#valence_model = makeValenceModel()
#genre_model = RNNGenreModel()
#regression_model = makeRegressionModel(arousal_model, valence_model, genre_model)
#trainAndTestGenre(input_file = 'audios_record.txt',filename='genres_record_num.txt',model = genre_model, name = "genre_model")
#trainAndTest(input_file = 'audios_record.txt',filename='arousal.txt',model = arousal_model, name = "arousal_model")
#trainAndTest(input_file = 'audios_record.txt',filename='valence.txt',model = valence_model, name = "valence_model")


#print(valence_model.summary())

#print(arousal_model.summary())

#print(genre_model.summary())
#val_lin, aro_lin = makeRegressionModel(valence_model,arousal_model,genre_model)

def getSongValenceArousalData(songfile):
    valence_model = load_model('valence_model')
    arousal_model = load_model('arousal_model')
    genre_model = load_model('genre_model')
    val_lin, aro_lin = joblib.load("valence_linear_model.pkl"),joblib.load("arousal_linear_model.pkl")
    valence_trends, arousal_trends = getTrends(songfile, valence_model, arousal_model, genre_model, val_lin, aro_lin)
    valence = np.mean(valence_trends)
    arousal = np.mean(arousal_trends)
    return valence, arousal, valence_trends, arousal_trends

if __name__ == "__main__":
    a,b,c,d = getSongValenceArousalData('5.wav')
    print(a)
    print(b)
    print(c)
    print(d)

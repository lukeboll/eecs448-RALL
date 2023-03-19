#Importing all the necessary packages
import nltk
import librosa
import torch
import gradio as gr
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
import python_speech_features
from sklearn import preprocessing
from python_speech_features import mfcc
import numpy as np
import os
import csv
from sklearn import preprocessing
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

genre_tags = ['ROCK', 'JAZZ', 'POP', 'ELEC', 'WORLD','COUNTRY']
full_data_len = 2000

def load_data(input_file):
  #reading the file
  aud, sample_rate = librosa.load(input_file)
  #make it 1-D
  if len(aud.shape) > 1: 
      aud = aud[:,0] + aud[:,1]
  aud= librosa.resample(aud,orig_sr=sample_rate,target_sr=16000)
  sample_rate = 16000
  return aud,sample_rate

def splitAudio(filepath,t1=15,t2=25,t3=35):
    newAudio,rate = load_data(filepath)
    div1,div2,div3 = t1*rate, t2*rate, t3*rate
    newAudio = [newAudio[div1:div2], newAudio[div2:div3], newAudio[div3:]]
    return newAudio[0], newAudio[1], newAudio[2]
'''
def resampleFeature(f0,f1,f2,sr=100):
    s1,s2,s3=[0],[0],[0]
    i=sr
    new_len = int(len(f0)/sr)
    while i<len(f0):
        s1.append(sum(f0[i-sr:i]) / sr)
        i += sr
    i=sr
    while i<len(f1):
        s2.append(sum(f1[i-sr:i]) / sr)
        i += sr
    i=sr
    while i<len(f2):
        s3.append(sum(f2[i-sr:i]) / sr)
        i += sr
    while len(s1)<new_len:
        s1.append(0)
    while len(s2)<new_len:
        s2.append(0)
    while len(s3)<new_len:
        s3.append(0)
    return s1,s2,s3
'''
def getRawFeatures(audios, sr=100):
   #feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
   feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=sr, 
                                             padding_value=0.0, do_normalize=True, return_attention_mask=False)
   return feature_extractor(audios,sampling_rate=sr, max_length = 16*sr)['input_values']
def genMFCCFeatures(signal, samplerate=16000):
    mfcc_features = mfcc(signal,
                         samplerate,
                         winlen = 0.05,
                         winstep = 0.01,
                         numcep = 50,
                         nfft = 1024)
    mfcc_features = preprocessing.scale(mfcc_features)
    return mfcc_features

def genTrainData(raw_testing_data, data_len = 2000):
    audios = []
    genres = []
    id = []
    for i in range(data_len):
        if i%100 == 0:
            print('processing '+str(i))
        filename = str(i)+".wav"
        if os. path. isfile(filename) and (str(i) in raw_testing_data):
            aud1, aud2, aud3 = splitAudio(filename)
            aud3 = aud3[:160000]
            #aud1, aud2, aud3 =  resampleFeature(aud1,aud2,aud3)
            audios.append(genMFCCFeatures(aud1))
            audios.append(genMFCCFeatures(aud2))
            audios.append(genMFCCFeatures(aud3))
            g = raw_testing_data[str(i)]
            genres.append(g)
            genres.append(g)
            genres.append(g)
            id.append(i)
            id.append(i)
            id.append(i)
    return audios, genres,id

def toTag(str):
    for tag in genre_tags:
        if tag in (str.upper()):
            return tag
    return 'Unknown'

def getTesting(filename):
    ret = {}
    with open(filename,'r',encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if(len(row)==0 or row[0]=='1298' or row[0]=='1814'):
                continue
            if toTag(row[2]) != 'Unknown':
                ret[row[0]] = toTag(row[2])
    return ret
# Rock, Jazz==JAZZ, Pop, Electronic==Electro, World==World Musi
# ROCK, JAZZ, POP, ELEC, WORLD
# 0 1 2 3 4

def helperAnalyzeData(data):
    counter = {'ROCK':0, 'JAZZ':0, 'POP':0, 'ELEC':0, 'WORLD':0, 'COUNTRY':0}
    all = 0
    for tag in data:
        g = data[tag].upper()
        for genre in counter:
            if genre in g:
                counter[genre] += 1
                all += 1
                break
    print(all)
    return counter

def dataGeneration():


    raw_testing_data = getTesting('DEAM_names.csv')
    #print(helperAnalyzeData(raw_testing_data))
    audios, genres, id = genTrainData(raw_testing_data)

    with open('genres_record.txt', 'w',encoding='utf-8') as f:
        for k in genres:
            f.write(k+'\n')

    with open('audios_record.txt', 'w',encoding='utf-8') as ff:
        cnt = 0
        for a in audios:
            ff.write('AUD '+str(cnt)+'\n')
            for dat in a:
                for num in dat:
                    ff.write(str(num)+' ')
                ff.write('\n')
            cnt += 1
        ff.write('AUD END')
    with open('id_record.txt', 'w',encoding='utf-8') as f:
        for k in id:
            f.write(str(k)+'\n')

    print("finished writing!")

#dataGeneration()

def peek(filename, lim = 100):
    with open(filename, 'r', encoding='utf-8') as f:
        cnt = 0
        total = []
        one = 0
        for s in f:
            #print(s)
            one += 1
            if s[0]=='A':
                cnt += 1
                total.append(one)
                one  = 0
            if cnt>= lim:
                break
    print(total)

def makeRawData(f_X, f_y,lim=2010):
    X = []
    y = []
    tags = genre_tags
    with open(f_X,'r',encoding='utf-8') as fx:
        cnt_group = 0
        sav = []
        flg = True
        for s in fx:
            if cnt_group >= lim:
                break
            if flg:
                flg = False
                continue # skip the first line
            if s[0]=='A':
                X.append(sav.copy())
                sav = []
                cnt_group += 1
            else:
                s=s.split(' ')
                cur_sav = []
                for w in s:
                    if len(w) <=1:
                        continue
                    cur_sav.append(float(w))
                sav.append(cur_sav.copy())
    with open(f_y, 'r', encoding='utf-8') as fy:
        for s in fy:
            flg = False
            for i in range(len(tags)):
                if s[:-1] == tags[i]:#remove the \n
                    y.append(i)
                    flg = True
            if not flg:
                y.append(-1)
                print(s)
    return X,y

def makeGenreData(X):
    newX = np.zeros((len(X[0][0]),len(X),len(X[0])))
    for i in range(len(X)):
        for j in range(len(X[i])):
            for k in range(len(X[i][j])):
                newX[k][i][j] = X[i][j][k]
    return newX

X,y = makeRawData('audios_record.txt','genres_record.txt')

data_size = len(X)
testing_size = int(data_size*0.2)
training_size = data_size - testing_size

trainX, testX, trainY, testY = train_test_split(X,y,test_size=testing_size,train_size=training_size)
#trainX, testX, trainY, testY = np.array(trainX), np.array(testX), np.array(trainY), np.array(testY)
trainX, testX = makeGenreData(trainX), makeGenreData(testX)
#print(len(trainX))
feature_len = len(trainX)
forest_model = []
for i in range(len(trainX)):
    print("fitting random forest on feature "+str(i))
    tree_model = RandomForestClassifier(max_depth=100).fit(trainX[i,:,:], trainY)
    forest_model.append(tree_model)

def getPrediction(l):
    lar = 0
    ret = -1
    for i in range(len(l)):
        if l[i]>lar:
            lar = l[i]
            ret = i
    return ret
def weightFunction(x):
    return x**0.75
def testAccuracy(forest_model,testX, testY):
    pred_rec = np.zeros((len(testY), len(genre_tags)))
    for i in range(feature_len):
        cur_pred = forest_model[i].predict(testX[i,:,:])
        for j in range(len(testY)):
            pred_rec[j][cur_pred[j]] += weightFunction(i)
    cnt = 0
    for i in range(len(testY)):
        if testY[i] == getPrediction(pred_rec[i]):
            cnt += 1
    with open('sav_results.txt', 'w',encoding='utf-8') as f:
        for i in range(len(testY)):
            f.write(str(pred_rec[i])+ ' ' + str(testY[i]) + '\n')
        for i in range(len(testY)):
            c = ""
            for j in range(feature_len):
                if pred_rec[i][j] == testY[i]:
                   c = c+" " + str(j)
            f.write(c+"\n")
    return cnt / len(testY)

X = makeGenreData(X)
print(testAccuracy(forest_model,X,y))
print(testAccuracy(forest_model,testX,testY))

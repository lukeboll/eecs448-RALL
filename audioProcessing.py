import python_speech_features
import scipy
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from python_speech_features import mfcc
import numpy as np
import os
import tqdm
from sklearn import preprocessing
import ffmpeg
from pydub import AudioSegment
import subprocess
from pydub.silence import split_on_silence
import torch
import torchaudio
import IPython
import requests
import acrcloud
import csv
# use python_speech_features for MFCC
def genMFCC(filepath):
    samplerate, signal = read(filepath)
    mfcc_features = mfcc(signal,
                         samplerate,
                         winlen = 0.05,
                         winstep = 0.01,
                         numcep = 10,
                         nfft = 1024)
    mfcc_features = preprocessing.scale(mfcc_features)
    return mfcc_features

def splitAudio(filepath, id, t1, t2):
    newAudio = AudioSegment.from_wav(filepath)
    newAudio = newAudio[t1:t2]
    newAudio.export(str(id)+'.wav', format="wav") #Exports to a wav file in the current path.

def loadAudio(filepath):
    newAudio = AudioSegment.from_wav(filepath)
    print("Done!")

def getName(filename):
    config = {
	    "key": "7dcf14702f55380d375bd9c62904bb1a",
	    "secret": "m00m5kbIXvPxaupxYfFu8MWtPrqeUZMKbnwyLFt3",
	    "host": "https://identify-us-west-2.acrcloud.com"
    }
    audio = filename
    acr = acrcloud.ACRcloud(config)
    name = acr.recognizer(audio)
    return name

def getNames(len = 2000):
    with open('DEAM_names.csv', 'w',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['artist','genre','spotify_id'])
        for i in range(2,len):
            print(i)
            filename = str(i)+".wav"
            if os. path. isfile(filename):
                data = getName(filename)
                if data['status']['msg'] == 'No result':
                    continue
                if not ('genres' in data['metadata']['music'][0]):
                    continue
                savid = ""
                if 'spotify' in data['metadata']['music'][0]['external_metadata']:
                    savid = data['metadata']['music'][0]['external_metadata']['spotify']['track']['id']
                #print(data['metadata']['music'][0])
                print("written")
                writer.writerow(
                    [data['metadata']['music'][0]['artists'][0]['name'],
                     data['metadata']['music'][0]['genres'][0]['name'][:10],
                     savid
                     ])
def str_to_val(s):
    if s[0] != '-':
        return float(s)
    return -float(s[1:])
def read_csv(filename, type, restrict = 2000):
    cnt =0
    all_set = []
    with open('id_record.txt','r',encoding='utf-8') as f0:
        for ord in f0:
            all_set.append(int(ord))
    with open(type+'.txt', 'w',encoding='utf-8') as f:
        with open(filename,'r',encoding='utf-8') as source:
            for row in source:
                if cnt>=restrict:
                    break
                cnt += 1
                if cnt==1:
                    continue
                data = row.split(',')
                for i in range(len(data)):
                    data[i] = str_to_val(data[i])
                data[0] = int(data[0]) # the id
                if not data[0] in all_set:
                    continue
                values = [sum(data[2:22])/10, sum(data[22:42])/10, sum(data[42:62])/10]
                #output = str(data[0]) + ' '+str(values[0]) + ' ' + str(values[1]) + ' '+ str(values[2])
                #output = output + '\n'
                #f.write(output)
                f.write(str(values[0])+'\n')
                f.write(str(values[1])+'\n')
                f.write(str(values[2])+'\n')
    pass
#print("test")
#cur_mfcc = genMFCC('test-real.wav')
#splitAudio('test-real.wav',0,15)
#loadAudio('fortest.wav')
#print(genMFCC('fortest.wav'))
#print(getName('fortest3.wav')['status']['msg'])
#getNames(len)
read_csv('arousal.csv','arousal')
read_csv('valence.csv','valence')

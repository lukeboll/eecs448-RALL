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

def getNames(len):
    print(len)
    with open('DEAM_names.csv', 'w',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id','artist','genre','spotify_id'])
        for i in range(len):
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
                    [str(i),
                     data['metadata']['music'][0]['artists'][0]['name'],
                     data['metadata']['music'][0]['genres'][0]['name'][:10],
                     savid
                     ])
#print("test")
#cur_mfcc = genMFCC('test-real.wav')
#splitAudio('test-real.wav',0,15)
#loadAudio('fortest.wav')
#print(genMFCC('fortest.wav'))
#print(getName('fortest3.wav')['status']['msg'])
getNames(2010)

# Common libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Linear regression Question libraries
from sklearn.linear_model import LinearRegression

# Logistic Regression Question Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import seaborn as sns


nltk.download('punkt')
nltk.download('stopwords')

# Reading in the Spotify Data
df = pd.read_csv('spotify_songs.csv')

new_df = pd.DataFrame({
    'text': df.loc[:, 'lyrics'],
    'label': df.loc[:, 'track_popularity']
})

nltk_stopwords = set(stopwords.words('english'))

def PreprocessSentence(sentence, nltk_stopwords):
    s = sentence.lower()
    s = word_tokenize(s)
    words = []
    for w in s:
        if w not in nltk_stopwords and len(w) > 1:
            words.append(w)
    return ' '.join(words)

def PreprocessData(new_df, nltk_stopwords):
    # Input - the dataframe, with columns label and text
    # output - the dataframe with the text processed as described earlier
    new_df.dropna(subset=['text'], inplace=True)  # drop rows with missing values in the 'text' column
    new_df.text = new_df.text.map(lambda x: PreprocessSentence(x, nltk_stopwords))
    return new_df

new_df = PreprocessData(new_df, nltk_stopwords)

print(new_df.text)

import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import langdetect
import numpy as np
import re


nltk.download('stopwords')
nltk_stopwords = set(stopwords.words('english'))

# Remove the URLS
def filter_websites(text):
    pattern = r'(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z][a-z\-]*'
    text = re.sub(pattern, '', text)
    if text.strip() == '':
        return np.nan
    else:
        return text

# Function to clean and process the sentence
def pre_process_sentence(sentence, nltk_stopwords):
    s = sentence.lower()
    s = s.split()
    words = []
    for w in s:
        w = w.strip(string.punctuation)
        if w not in nltk_stopwords and len(w) > 1:
            words.append(w)
    return ' '.join(words)

# Function to clean the dataframe of non-English lyrics
def is_english(text):
    try:
        return langdetect.detect(text) == 'en'
    except:
        return False

# Function to clean and process the dataframe
def pre_process_data(new_df, nltk_stopwords):
    new_df.dropna(subset=['text'], inplace=True)  # drop rows with missing values in the 'text' column
    new_df.text = new_df.text.map(lambda x: filter_websites(x))
    new_df.dropna(subset=['text'], inplace=True)  # drop rows with missing values in the 'text' column
    new_df = new_df[new_df['text'].apply(is_english)]
    new_df.text = new_df.text.map(lambda x: pre_process_sentence(x, nltk_stopwords))
    new_df.reset_index(drop=True, inplace=True)
    return new_df

if __name__ == "__main__":
    # Load data from CSV file
    df = pd.read_csv('spotify_songs.csv')

    # Rename columns and assign to new dataframe
    df = pd.DataFrame({
        'text': df.loc[:, 'lyrics'],
        'label': df.loc[:, 'track_popularity']
    })

    # Preprocess the dataframe
    df = pre_process_data(df, nltk_stopwords)

    # Save data to file
    df.to_csv('processed_spotify_data.csv', index=False)
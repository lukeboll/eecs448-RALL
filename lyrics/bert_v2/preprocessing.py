import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
import langdetect
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel


nltk.download('stopwords')
nltk_stopwords = set(stopwords.words('english'))

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
    new_df = new_df[new_df['text'].apply(is_english)]
    new_df.text = new_df.text.map(lambda x: pre_process_sentence(x, nltk_stopwords))
    new_df.reset_index(drop=True, inplace=True)
    return new_df

def get_bert_features(texts, tokenizer, model, max_len=512):
    input_ids = []
    attention_masks = []
    features = []
    
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,                     
                            add_special_tokens = True, 
                            max_length = max_len,      
                            padding='max_length',
                            truncation=True,
                            return_attention_mask = True,
                            return_tensors = 'pt',
                            )
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)

    with torch.no_grad():
        model_output = model(input_ids, attention_mask=attention_masks)
    
    last_hidden_states = model_output[0]
    features = last_hidden_states[:,0,:].numpy()

    return input_ids, attention_masks, features

if __name__ == "__main__":
    # Load data from CSV file
    df = pd.read_csv('spotify_songs.csv')

    # Rename columns and assign to new dataframe
    df = pd.DataFrame({
        'text': df.loc[:, 'lyrics'],
        'label': df.loc[:, 'track_popularity'],
        'genre': df.loc[:, 'playlist_genre']
    })

    # Remove genres we aren't looking at
    genres_to_keep = ['rock', 'pop', 'edm']
    df = df[df['genre'].isin(genres_to_keep)].reset_index(drop=True)

    # Preprocess the dataframe
    df = pre_process_data(df, nltk_stopwords)

    # Use BERT for feature extraction
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    input_ids, attention_masks, features = get_bert_features(df['text'], tokenizer, model)
    df['features'] = list(features)

    # Drop columns
    df.drop(columns=['text', 'features', 'input_ids', 'attention_mask'], inplace=True)

    # Save the datasets to their respective genres
    for genre in genres_to_keep:
        df_filtered = df[df['genre'] == genre].reset_index(drop=True)
        df_filtered = df_filtered.drop('genre', axis=1)
        df_filtered.to_csv(f"{genre}/processed_spotify_data.csv", index=False)

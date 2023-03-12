# Importing all libraries
import pandas as pd
import numpy as np

def testset_generation(output_filename):
    val_arousal = pd.read_csv("muse_v3.csv")
    val_arousal = val_arousal[['track', 'artist', 'valence_tags', 'arousal_tags', 'dominance_tags', 'genre']]
    val_arousal['track'] = val_arousal['track'].str.lower()
    val_arousal['artist'] = val_arousal['artist'].str.lower()
    df = pd.read_csv("tracks.csv")
    df = df[['name', 'popularity', 'duration_ms', 'artists', 'danceability', 'energy', 'key',
           'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
           'liveness', 'valence', 'tempo']]
    df['name'] = df['name'].str.lower()
    df = df[df[['name', 'artists']].duplicated(keep = False) == False]
    df = df[df['name'].isin(val_arousal['track'])]

    def transform_row(row):
        output = ""
        for i in eval(row):
            output += i.lower() + " & "
        return output[:-3]
    df['artists'] = df['artists'].apply(transform_row)

    val_arousal.set_index(pd.MultiIndex.from_frame(val_arousal[['track', 'artist']]), inplace = True)
    val_arousal = val_arousal[['valence_tags', 'arousal_tags', 'dominance_tags', 'genre']]
    merged_df = df.join(val_arousal, on = ['name', 'artists'], how = 'inner')
    merged_df.to_csv(output_filename)
    
testset_generation("prediction_dataset.csv")
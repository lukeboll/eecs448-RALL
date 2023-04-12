import pandas as pd
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import fuzz

# Read in the original dataframe
df = pd.read_csv('../../data/csv/prediction_dataset.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# Sort the dataframe by the desired column (e.g. "Name")
df = df.sort_values('artists', ascending=False)
df = df.reset_index(drop=True)

# Read in the labeled lyrics dataset and preprocess the artist and song names
prev_data = pd.read_csv("labeled_lyrics_cleaned.csv", on_bad_lines="skip")
prev_data['artist_processed'] = prev_data['artist'].apply(lambda x: x.lower().strip())
prev_data['song_processed'] = prev_data['song'].apply(lambda x: x.lower().strip())

# Loop through each row of the sorted dataframe
new_data_list = []
prev_letter = ''
no_letters = 0
file_not_found = 0
no_song_match = 0
no_song_but_artist_match = 0

for index, row in tqdm(df.iterrows(), total=df.shape[0]):

    # Preprocess the artist and song names for fuzzy matching
    artist_processed = row['artists'].lower().strip()
    song_processed = row['name'].lower().strip()

    # Search for matches in the prev_data based on the two columns you want to match on
    match = prev_data[(prev_data['artist_processed'].apply(lambda x: fuzz.token_set_ratio(x, artist_processed)) >= 80) & 
                      (prev_data['song_processed'].apply(lambda x: fuzz.token_set_ratio(x, song_processed)) >= 80)]

    if not match.empty:
        best_match = match.loc[match['song_processed'].apply(lambda x: fuzz.token_set_ratio(x, song_processed)).idxmax()]
        new_data_list.append(best_match['seq'])
    else:
        new_data_list.append(np.nan)
        no_song_match += 1

# Merge the new_data list with the original dataframe
df["lyrics"] = new_data_list

# Replace NaN values with empty strings, if desired
df = df.dropna()
df = df.reset_index(drop=True)
df.to_csv('lyrics_and_predictions_fourth_approach.csv', index=False)

print(f"No letter found: {no_letters}")
print(f"File not found: {file_not_found}")
print(f"No song match found: {no_song_match}")

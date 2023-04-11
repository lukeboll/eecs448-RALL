import pandas as pd
import numpy as np
from tqdm import tqdm

# Read in the original dataframe
df = pd.read_csv('../../data/csv/prediction_dataset.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

# Sort the dataframe by the desired column (e.g. "Name")
df = df.sort_values('artists', ascending=False)
df = df.reset_index(drop=True)

# Loop through each row of the sorted dataframe
new_data_list = []
prev_letter = ''
prev_data = None
no_letters = 0
file_not_found = 0
no_song_match = 0
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    # Determine the first valid letter or number in the column you sorted on
    letter = None
    for char in row['artists']:
        if char.isalnum():
            letter = char
            break
    
    if not letter:
        new_data_list.append(np.nan)
        no_letters += 1
        continue
    
    # If it's a number, set letter to "19" instead
    if letter.isdigit():
        letter = "19"

    # Load in the corresponding CSV file if it's not the same as the previous one
    if letter != prev_letter:
        csv_filename = f"../../data/csv/azlyrics-scraper/azlyrics_lyrics_{letter}.csv"
        try:
            prev_data = pd.read_csv(csv_filename, on_bad_lines="skip", usecols=['ARTIST_NAME', 'SONG_NAME', 'LYRICS'])

            # Remove rows with missing values
            prev_data.dropna(inplace=True)
            prev_letter = letter
        except FileNotFoundError:
            new_data_list.append(np.nan)
            file_not_found += 1
            continue
        

    # Search for matches in the new_data based on the two columns you want to match on
    match = prev_data.loc[(row['artists'] == prev_data['ARTIST_NAME']) & (row['name'] == prev_data['SONG_NAME'])]

    # If a match is found, add the corresponding entry from prev_data to a list
    if not match.empty:
        new_data_list.append(match.iloc[0]["LYRICS"])
    # If no match is found, add a row of NaN values to the list
    else:
        no_song_match += 1
        new_data_list.append(np.nan)

# Merge the new_data list with the original dataframe
df["lyrics"] = new_data_list

# Replace NaN values with empty strings, if desired
df = df.dropna()
df = df.reset_index(drop=True)
df.to_csv('lyrics_and_predictions.csv', index=False)

print(f"No letter found: {no_letters}")
print(f"File not found: {file_not_found}")
print(f"No song match found: {no_song_match}")
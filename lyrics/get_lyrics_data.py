import pandas as pd
import glob
from tqdm import tqdm
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from multiprocessing import Pool, cpu_count

# use glob to get all the csv files in the folder
folder_path = '../data/csv/azlyrics-scraper'
csv_files = glob.glob(folder_path + "/*.csv")

data_frames = []
for file in csv_files:
    df = pd.read_csv(file, error_bad_lines=False)
    data_frames.append(df)

lyrics = pd.concat(data_frames, ignore_index=True)

# Load the second data frame with song titles and artist
df2 = pd.read_csv("../prediction_dataset.csv")

# Define a function to perform fuzzy matching on song titles and artist names
def fuzzy_match(title1, artist1, title2, artist2):
    # First, compare the artist names
    artist_score = fuzz.token_set_ratio(artist1.lower(), artist2.lower())
    # If the artist names match, compare the song titles
    if artist_score > 80:
        title_score = fuzz.token_set_ratio(title1.lower(), title2.lower())
        if title_score > 80:
            return True
    return False

# Add a column to df2 to store the matched lyrics
df2["Lyrics"] = ""

# Define a function to match lyrics for a single row
def match_lyrics(row):
    for index, row2 in lyrics.iterrows():
        if fuzzy_match(str(row2["SONG_NAME"]), str(row2["ARTIST_NAME"]), str(row["name"]), str(row["artists"])):
            # If there is a match, copy the lyrics to df2
            return row2["LYRICS"]
    return ""

# Use multiprocessing to match lyrics for all rows in df2
def process_df2(df):
    with Pool(cpu_count()) as p:
        df["Lyrics"] = p.map(match_lyrics, df.to_dict("records"))
    return df

# Process df2 with multiprocessing
df2 = process_df2(df2)

# Save the results to a new CSV file
df2.to_csv("lyrical.csv", index=False)

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

### Data Preprocessing ###

# Load the data
valence_arousal_data = pd.read_csv('data/csv/muse_v3.csv', header=0)
valence_arousal_data = valence_arousal_data[['track', 'valence_tags', 'arousal_tags']]
valence_arousal_data = valence_arousal_data[valence_arousal_data['track'].notnull()]
valence_arousal_data = valence_arousal_data[valence_arousal_data['valence_tags'].notnull()]
valence_arousal_data = valence_arousal_data[valence_arousal_data['arousal_tags'].notnull()]

spotify_data = pd.read_csv('data/csv/song_data.csv', header=0)
spotify_data = spotify_data[['song_name', 'song_popularity']]

merged_df = pd.merge(valence_arousal_data, spotify_data, left_on='track', right_on='song_name', how='inner')
merged_df.drop('song_name', axis=1, inplace=True)
merged_df = merged_df.drop_duplicates(subset=['track'])
merged_df.drop('track', axis=1, inplace=True)
merged_df = merged_df.rename(columns={'valence_tags': 'valence', 'arousal_tags': 'arousal' , 'song_popularity': 'popularity'})

print(merged_df)
 
X = merged_df[['valence', 'arousal']].values
y = merged_df['popularity'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=485)

### Build Model ###

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Save the model
# model.save('data/models/popularity_model.h5')

### Testing ###

# Load the model
# model = tf.keras.models.load_model('data/models/popularity_model.h5')

# Evaluate the model on the testing set
mse, _ = model.evaluate(X_test, y_test)

# Print the mean squared error
print("Mean Squared Error:", mse)

# Generate some test data
test_data = pd.DataFrame({'valence': [0.6, 0.2, 0.8], 'arousal': [0.4, 0.9, 0.5]})
X_test = test_data.values

# Make predictions
y_pred = model.predict(X_test)

# Print the predictions
print("Predictions:", y_pred)

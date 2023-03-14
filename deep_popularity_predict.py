import tensorflow as tf
import pandas as pd
from  keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load the data
# valence_arousal_data = pd.read_csv('data/csv/muse_v3.csv', header=0)
# valence_arousal_data = valence_arousal_data[['track', 'valence_tags', 'arousal_tags']]
# valence_arousal_data = valence_arousal_data[valence_arousal_data['track'].notnull()]
# valence_arousal_data = valence_arousal_data[valence_arousal_data['valence_tags'].notnull()]
# valence_arousal_data = valence_arousal_data[valence_arousal_data['arousal_tags'].notnull()]

# spotify_data = pd.read_csv('data/csv/song_data.csv', header=0)
# spotify_data = spotify_data[['song_name', 'song_popularity']]

# merged_df = pd.merge(valence_arousal_data, spotify_data, left_on='track', right_on='song_name', how='inner')
# merged_df.drop('song_name', axis=1, inplace=True)
# merged_df = merged_df.drop_duplicates(subset=['track'])
# merged_df.drop('track', axis=1, inplace=True)
# merged_df = merged_df.rename(columns={'valence_tags': 'valence', 'arousal_tags': 'arousal' , 'song_popularity': 'popularity'})

# print(merged_df)
merged_df = pd.read_csv('data/csv/prediction_dataset.csv')
 
X = merged_df[['valence_tags', 'arousal_tags']].values
y = merged_df['popularity'].values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=485)

# Define the input layer
inputs = Input(shape=(2,))

# Define the hidden layers with dropout layers
x = Dense(64, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)

# Define the output layer
outputs = Dense(1, activation='sigmoid')(x)

# Define the model
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

# Save the model
# model.save('data/models/popularity_model.h5')

### Testing ###

# Load the model
# model = tf.keras.models.load_model('data/models/popularity_model.h5')

# Evaluate the model
test_loss = model.evaluate(x_test, y_test)

# Make predictions
predictions = model.predict(x_test)

print(test_loss)
print(predictions)

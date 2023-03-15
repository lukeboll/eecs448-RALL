import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

NUM_FEATURES = 11 # 10 input features, 1 output feature

DATA = pd.read_csv('data/csv/prediction_dataset.csv', header=0)

DATA = DATA[DATA['popularity'].notnull()]
DATA = DATA[DATA['danceability'].notnull()]
DATA = DATA[DATA['energy'].notnull()]
DATA = DATA[DATA['loudness'].notnull()]
DATA = DATA[DATA['speechiness'].notnull()]
DATA = DATA[DATA['acousticness'].notnull()]
DATA = DATA[DATA['instrumentalness'].notnull()]
DATA = DATA[DATA['liveness'].notnull()]
DATA = DATA[DATA['valence_tags'].notnull()]
DATA = DATA[DATA['arousal_tags'].notnull()]
DATA = DATA[DATA['genre'].notnull()]

def get_genre_data(genre): # rock, pop, indie, soul, folk
    genre_data = DATA[DATA['genre'] == genre]
    X = genre_data[[
        'danceability', 
        'energy', 
        'loudness', 
        'speechiness',
        'acousticness', 
        'instrumentalness', 
        'liveness',
        'valence_tags', 
        'arousal_tags'
    ]].astype(float).values
    y = genre_data['popularity'].astype(float).values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=485)

    return X_train, X_test, y_train, y_test

X_train_rock, X_test_rock, y_train_rock, y_test_rock = get_genre_data('rock')
X_train_pop, X_test_pop, y_train_pop, y_test_pop = get_genre_data('pop')
X_train_indie, X_test_indie, y_train_indie, y_test_indie = get_genre_data('indie')
X_train_soul, X_test_soul, y_train_soul, y_test_soul = get_genre_data('soul')
X_train_folk, X_test_folk, y_train_folk, y_test_folk = get_genre_data('folk')

# Build the model
model = Sequential([
    Dense(64, input_shape=(9,), activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])


# Train the models
rock_history = model.fit(X_train_rock, y_train_rock, validation_split=0.1, batch_size=32, epochs=100)
pop_history = model.fit(X_train_pop, y_train_pop, validation_split=0.1, batch_size=32, epochs=100)
indie_history = model.fit(X_train_indie, y_train_indie, validation_split=0.1, batch_size=32, epochs=100)
soul_history = model.fit(X_train_soul, y_train_soul, validation_split=0.1, batch_size=32, epochs=100)
folk_history = model.fit(X_train_folk, y_train_folk, validation_split=0.1, batch_size=32, epochs=100)


# Save the model
# model.save('data/models/popularity_model.h5')

### Testing ###

# Load the model
# model = tf.keras.models.load_model('data/models/popularity_model.h5')

# Evaluate the model
test_loss_rock, test_mae_rock = model.evaluate(X_test_rock, y_test_rock)
test_loss_pop, test_mae_pop = model.evaluate(X_test_pop, y_test_pop)
test_loss_indie, test_mae_indie = model.evaluate(X_test_indie, y_test_indie)
test_loss_soul, test_mae_soul = model.evaluate(X_test_soul, y_test_soul)
test_loss_folk, test_mae_folk = model.evaluate(X_test_folk, y_test_folk)

print(f'Test MAE (ROCK): {test_mae_rock:.2f}')
print(f'Test MAE (POP): {test_mae_pop:.2f}')
print(f'Test MAE (INDIE): {test_mae_indie:.2f}')
print(f'Test MAE (SOUL): {test_mae_soul:.2f}')
print(f'Test MAE (FOLK): {test_mae_folk:.2f}')

# Make predictions
# predictions = model.predict(X_test_rock)


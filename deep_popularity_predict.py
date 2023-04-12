import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

NUM_FEATURES = 11 # 10 input features, 1 output feature

DATA = pd.read_csv('data/csv/prediction_dataset.csv', header=0)

DATA = DATA[DATA['genre'].notnull()]
DATA = DATA[DATA['valence'].notnull()]
DATA = DATA[DATA['arousal'].notnull()]
DATA = DATA[DATA['bert_features'].notnull()]
DATA = DATA[DATA['popularity'].notnull()]



def get_genre_data(genre): # rock, pop, indie, soul, folk
    genre_data = DATA[DATA['genre'] == genre]
    X = genre_data[[
        'valence', 
        'arousal', 
        'bert_features'
    ]].astype(float).values
    y = genre_data['popularity'].astype(float).values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=448)

    return X_train, X_test, y_train, y_test


X_train_pop, X_test_pop, y_train_pop, y_test_pop = get_genre_data('pop')
X_train_rock, X_test_rock, y_train_rock, y_test_rock = get_genre_data('rock')
X_train_edm, X_test_edm, y_train_edm, y_test_edm = get_genre_data('edm')


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
pop_model = model.fit(X_train_pop, y_train_pop, validation_split=0.1, batch_size=32, epochs=100)
rock_model = model.fit(X_train_rock, y_train_rock, validation_split=0.1, batch_size=32, epochs=100)
edm_model = model.fit(X_train_edm, y_train_edm, validation_split=0.1, batch_size=32, epochs=100)


# Save the model
# pop_model.save('data/models/pop_popularity_model.h5')
# rock_model.save('data/models/rock_popularity_model.h5')
# edm_model.save('data/models/edm_popularity_model.h5')

### Testing ###

# Load the model
# pop_model = tf.keras.models.load_model('data/models/pop_popularity_model.h5')
# rock_model = tf.keras.models.load_model('data/models/rock_popularity_model.h5')
# edm_model = tf.keras.models.load_model('data/models/edm_popularity_model.h5')

# Evaluate the model
test_loss_pop, test_mae_pop = model.evaluate(X_test_pop, y_test_pop)
test_loss_rock, test_mae_rock = model.evaluate(X_test_rock, y_test_rock)
test_loss_edm, test_mae_edm = model.evaluate(X_test_edm, y_test_edm)

print(f'Test MAE (POP): {test_mae_pop:.2f}')
print(f'Test MAE (ROCK): {test_mae_rock:.2f}')
print(f'Test MAE (EDM): {test_mae_edm:.2f}')

# Make predictions
# pop_prediction = model.predict(X_test_pop)
# rock_prediction = model.predict(X_test_rock)
# edm_prediction = model.predict(X_test_edm)


import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

### Data Preprocessing ###

# Load the data
# TODO: replace 'valence-arousal.csv' with valence and arousal predictions
data = pd.read_csv('data/csv/valence-arousal.csv') 
X = data[['valence', 'arousal']].values
y = data['popularity'].values

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
model.save('data/models/popularity_model.h5') # replace 'popularity_model.h5' with your desired filename

### Testing ###

# Load the model
model = tf.keras.models.load_model('data/models/popularity_model.h5')

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

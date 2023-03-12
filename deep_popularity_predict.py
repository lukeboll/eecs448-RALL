import tensorflow as tf
import pandas as pd
from  keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load the data
# TODO: replace 'valence-arousal.csv' with valence and arousal predictions
data = pd.read_csv('data/csv/valence-arousal.csv')
 
X = data[['valence', 'arousal']].values
y = data['popularity'].values

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
history = model.fit(x_train, y_train, epochs=100, validation_split=0.2)

# Save the model
# model.save('data/models/popularity_model.h5')

### Testing ###

# Load the model
# model = tf.keras.models.load_model('data/models/popularity_model.h5')

# Evaluate the model
test_loss = model.evaluate(x_test, y_test)

# Make predictions
predictions = model.predict(x_test)

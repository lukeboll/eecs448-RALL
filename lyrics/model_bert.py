import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from bert import tokenization
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load pre-trained BERT model
module_url = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True, signature="tokens")

# Load data from CSV file
data = pd.read_csv("spotify_songs.csv")

# Split the DataFrame into train and test sets
train_df, test_df = train_test_split(data, test_size=0.2, random_state=448)

# Define the tokenizer
FullTokenizer = tokenization.FullTokenizer

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy() #The vocab file of bert for tokenizer

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case) 

# Tokenize the data
input_ids = []
input_masks = []
segment_ids = []

for text in train_df["lyrics"]:
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Add [CLS] and [SEP] tokens
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    # Convert tokens to input IDs, input masks, and segment IDs
    input_id = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_id)
    segment_id = [0] * len(input_id)

    # Pad or truncate the input to a fixed length
    max_seq_length = 128
    if len(input_id) > max_seq_length:
        input_id = input_id[:max_seq_length]
        input_mask = input_mask[:max_seq_length]
        segment_id = segment_id[:max_seq_length]
    else:
        padding = [0] * (max_seq_length - len(input_id))
        input_id += padding
        input_mask += padding
        segment_id += padding

    # Add the input IDs, input masks, and segment IDs to the lists
    input_ids.append(input_id)
    input_masks.append(input_mask)
    segment_ids.append(segment_id)

# Convert the lists to numpy arrays
input_ids = np.array(input_ids)
input_masks = np.array(input_masks)
segment_ids = np.array(segment_ids)

# Define the model
input_ids_layer = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_ids")
input_masks_layer = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_masks")
segment_ids_layer = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

bert_output = bert_layer([input_ids_layer, input_masks_layer, segment_ids_layer])

dropout_layer = tf.keras.layers.Dropout(0.2)(bert_output["pooled_output"])
output_layer = tf.keras.layers.Dense(1, activation="linear")(dropout_layer)

model = tf.keras.models.Model(inputs=[input_ids_layer, input_masks_layer, segment_ids_layer], outputs=output_layer)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss="mean_squared_error", metrics=["mae"])

# Train the model
model.fit([input_ids, input_masks, segment_ids], train_df["track_popularity"], epochs=5, batch_size=32, validation_split=0.2)

# Make predictions on new data
new_input_ids = []
new_input_masks = []
new_segment_ids = []

for text in test_df["lyrics"]:
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Add [CLS] and [
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    # Convert tokens to input IDs, input masks, and segment IDs
    input_id = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_id)
    segment_id = [0] * len(input_id)

    # Pad or truncate the input to a fixed length
    max_seq_length = 128
    if len(input_id) > max_seq_length:
        input_id = input_id[:max_seq_length]
        input_mask = input_mask[:max_seq_length]
        segment_id = segment_id[:max_seq_length]
    else:
        padding = [0] * (max_seq_length - len(input_id))
        input_id += padding
        input_mask += padding
        segment_id += padding

    # Add the input IDs, input masks, and segment IDs to the lists
    new_input_ids.append(input_id)
    new_input_masks.append(input_mask)
    new_segment_ids.append(segment_id)

new_input_ids = np.array(new_input_ids)
new_input_masks = np.array(new_input_masks)
new_segment_ids = np.array(new_segment_ids)

predictions = model.predict([new_input_ids, new_input_masks, new_segment_ids])
predictions_train = model.predict([input_ids, input_masks, segment_ids])

train_mse = mean_squared_error(train_df["track_popularity"], predictions_train)
test_mse = mean_squared_error(test_df["track_popularity"], predictions)


print(f"Training MSE: {train_mse}")
print(f"Testing MSE: {test_mse}")
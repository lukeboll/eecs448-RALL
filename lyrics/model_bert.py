import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('spotify_songs.csv')

# drop NaN values
df.dropna(inplace=True)

# reset the index
df.reset_index(drop=True, inplace=True)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

test_df.to_csv('test_data.csv', index=False)



# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Tokenize input data and convert to PyTorch tensors
input_ids = []
attention_masks = []
labels = []
for i in range(len(train_df)):
    encoded_dict = tokenizer.encode_plus(train_df['lyrics'][i],
                                          add_special_tokens=True,
                                          max_length=128,
                                          pad_to_max_length=True,
                                          return_attention_mask=True,
                                          return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    labels.append(torch.tensor(train_df['track_popularity'][i], dtype=torch.float))
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels, dtype=torch.float)

print("Data tokenized and converted")
# Create dataset and dataloader
dataset = TensorDataset(input_ids, attention_masks, labels)
train_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=32)


# Create dataloader for validation set
input_ids_test = []
attention_masks_test = []
labels_test = []
for i in range(len(test_df)):
    encoded_dict = tokenizer.encode_plus(val_df['lyrics'][i],
                                        add_special_tokens=True,
                                        max_length=128,
                                        pad_to_max_length=True,
                                        return_attention_mask=True,
                                        return_tensors='pt')
    input_ids_test.append(encoded_dict['input_ids'])
    attention_masks_test.append(encoded_dict['attention_mask'])
    labels_test.append(torch.tensor(test_df['track_popularity'][i], dtype=torch.float))
input_ids_test = torch.cat(input_ids_test, dim=0)
attention_masks_test = torch.cat(attention_masks_test, dim=0)
labels_test = torch.tensor(labels_test, dtype=torch.float)

dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
test_dataloader = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=32)

# Set hyperparameters
learning_rate = 2e-5

# Create optimizer and set learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

print("Training has started")

# Train BERT model
start = time.time()

curr_patience = 0
patience = 5
epoch = 0
training_loss = []
validation_loss = []
global_min_loss = float('inf')

while curr_patience < patience:
    timeTaken = time.time() - start
    print(f'Epoch {epoch + 1}. Time is {timeTaken:.4f}')
    model.train()
    train_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        batch_loss = criterion(outputs, labels)
        train_loss += batch_loss.item()
        batch_loss.backward()
        optimizer.step()
    train_loss /= len(train_dataloader)
    print(f'Training loss: {train_loss:.4f}')
    

    # Evaluate model on testing set
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            batch_loss = criterion(outputs, labels)
            val_loss += batch_loss.item()
        val_loss /= len(test_dataloader)
        print(f'Validation loss: {val_loss:.4f}')
        

    training_loss.append(train_loss)
    validation_loss.append(val_loss)

    if val_loss < global_min_loss:
        global_min_loss = val_loss
        curr_patience = 0
        model_save_name = "model_epoch_" + f"{epoch + 1}"
        torch.save(model.state_dict(), model_save_name)
    else:
        curr_patience += 1

plt.figure()

# Plot the data
plt.plot(training_loss, 'b-', label='Training Loss')
plt.plot(validation_loss, 'r-', label='Validation Loss')

# Add a legend
plt.legend()

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("BERT Training")
plt.savefig("BERT_model_training.png")
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
import time

# Load data from CSV file
df = pd.read_csv('spotify_songs.csv')

# drop NaN values
df.dropna(inplace=True)

# reset the index
df.reset_index(drop=True, inplace=True)

df, test_df = train_test_split(df, test_size=0.2, random_state=42)

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
for i in range(len(df)):
    encoded_dict = tokenizer.encode_plus(df['lyrics'][i],
                                          add_special_tokens=True,
                                          max_length=128,
                                          pad_to_max_length=True,
                                          return_attention_mask=True,
                                          return_tensors='pt')
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    labels.append(torch.tensor(df['track_popularity'][i], dtype=torch.float))
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.stack(labels)

print("Data tokenized and converted")
# Create dataset and dataloader
dataset = TensorDataset(input_ids, attention_masks, labels)
train_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=32)

# Set hyperparameters
epochs = 4
learning_rate = 2e-5

# Create optimizer and set learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Training has started")
# Train BERT model
start = time.time()
for epoch in range(epochs):
    timeTaken = time.time() - start
    print(f'Epoch {epoch + 1} of {epochs}. Time is {timeTaken:.4f}')
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(train_dataloader)
    print(f'Average training loss: {avg_loss}')

torch.save(model.state_dict(), 'my_model.pth')

# Create dataloader for testing set
input_ids_test = []
attention_masks_test = []
labels_test = []
for i in range(len(test_df)):
    encoded_dict = tokenizer.encode_plus(test_df['lyrics'][i],
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
labels_test = torch.stack(labels_test)

dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
test_dataloader = DataLoader(dataset_test, sampler=SequentialSampler(dataset_test), batch_size=32)

# Evaluate model on testing set
model.eval()
total_loss_test = 0
with torch.no_grad():
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss_test += loss.item()

avg_loss_test = total_loss_test / len(test_dataloader)
print(f'Average testing loss: {avg_loss_test}')
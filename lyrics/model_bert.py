import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

# Load data from CSV file
df = pd.read_csv('spotify_songs.csv')

# drop NaN values
df.dropna(inplace=True)

# reset the index
df.reset_index(drop=True, inplace=True)


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
for epoch in range(epochs):
    print(f'Epoch {epoch + 1} of {epochs}')
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
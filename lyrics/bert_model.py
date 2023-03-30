import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel, AdamW
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from train_common import *
from utils import config
import utils
import time


# Read in the data
data_path = "processed_spotify_data.csv"
df = pd.read_csv(data_path)
print(f"Full data size: {df.shape[0]}")

# Tokenize Data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_corpus = tokenizer(text=df.text.tolist(),
                            add_special_tokens=True,
                            padding='max_length',
                            truncation='longest_first',
                            max_length=300,
                            return_attention_mask=True)

input_ids = encoded_corpus['input_ids']
attention_mask = encoded_corpus['attention_mask']

# Filter long inputs
def filter_long_descriptions(tokenizer, descriptions, max_len):
    indices = []
    lengths = tokenizer(descriptions, padding=False, 
                     truncation=False, return_length=True)['length']
    for i in range(len(descriptions)):
        if lengths[i] <= max_len-2:
            indices.append(i)
    return indices

short_descriptions = filter_long_descriptions(tokenizer, 
                               df.text.tolist(), 300)
input_ids = np.array(input_ids)[short_descriptions]
attention_mask = np.array(attention_mask)[short_descriptions]
labels = df.label.to_numpy()[short_descriptions]

# Break up the data into train, validation, and testing sets
test_size = 0.2
seed = 448

train_inputs, test_inputs, old_train_labels, test_labels = \
            train_test_split(input_ids, labels, test_size=test_size, 
                             random_state=seed)
train_masks, test_masks, _, _ = train_test_split(attention_mask, 
                                        labels, test_size=test_size, 
                                        random_state=seed)

train_inputs, val_inputs, train_labels, val_labels = \
            train_test_split(train_inputs, old_train_labels, test_size=test_size, 
                             random_state=seed)
train_masks, val_masks, _, _ = train_test_split(train_masks, 
                                        old_train_labels, test_size=test_size, 
                                        random_state=seed)

# Standardize the target variable in training and validation
popularity_scaler = StandardScaler()
popularity_scaler.fit(train_labels.reshape(-1, 1))
train_labels = popularity_scaler.transform(train_labels.reshape(-1, 1))
val_labels = popularity_scaler.transform(val_labels.reshape(-1, 1))

# Make the dataloaders for PyTorch
batch_size = 32

def create_dataloaders(inputs, masks, labels, batch_size):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor, 
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True)
    return dataloader

tr_loader = create_dataloaders(train_inputs, train_masks, 
                                      train_labels, batch_size)
va_loader = create_dataloaders(val_inputs, val_masks, 
                                     val_labels, batch_size)

te_loader = create_dataloaders(test_inputs, test_masks, 
                                     test_labels, batch_size)

# Create the architecture and instance of the model
class BertRegressor(nn.Module):
    
    def __init__(self, drop_rate=0.2, freeze_camembert=False):
        super(BertRegressor, self).__init__()
        D_in, D_out = 768, 1
        
        self.bert = \
                   BertModel.from_pretrained("bert-base-uncased")
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))
        
    def forward(self, input_ids, attention_masks):
        outputs = self.bert(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs
    
model = BertRegressor(drop_rate=0.2)

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")
model.to(device)

# Set Up the Criterion and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                  lr=5e-5,
                  eps=1e-8)

print("Number of float-valued parameters:", count_parameters(model))

# Training Loop with Validation
# Attempts to restore the latest checkpoint if exists
print("Loading cnn...")
model, start_epoch, stats = restore_checkpoint(model, "./checkpoints/target/")

axes = utils.make_training_plot()

# Evaluate the randomly initialized model
evaluate_epoch(
    axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats, device
)

# initial val loss for early stopping
global_min_loss = stats[0][0]

# Define patience for early stopping. Replace "None" with the patience value.
patience = 5
curr_count_to_patience = 0

epoch = start_epoch
time0 = time.time()
while curr_count_to_patience < patience:
    print(f"Epoch Number: {epoch + 1}\nTime taken so far: {(time.time() - time0) / 3600} hours\n")
    # Train model
    train_epoch(tr_loader, model, criterion, optimizer, device)

    # Evaluate model
    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats
    )

    # Save model parameters
    save_checkpoint(model, epoch + 1, "./checkpoints/target/", stats)

    # update early stopping parameters
    curr_count_to_patience, global_min_loss = early_stopping(
        stats, curr_count_to_patience, global_min_loss
    )

    epoch += 1

print("Finished Training")

# Save figure and keep plot open
utils.save_cnn_training_plot()
utils.hold_training_plot()

print("Loading cnn...")
model, start_epoch, stats = restore_checkpoint(model, "./checkpoints/target/")

axes = utils.make_training_plot()

evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
        update_plot=False,
    )
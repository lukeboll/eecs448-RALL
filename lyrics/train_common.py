"""
EECS 445 - Introduction to Machine Learning
Winter 2023  - Project 2

Helper file for common training functions.
"""

from utils import config
import numpy as np
import itertools
import os
import torch
from torch.nn.functional import softmax
from torch.nn.utils.clip_grad import clip_grad_norm
from sklearn import metrics
import utils


def count_parameters(model):
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, epoch, checkpoint_dir, stats):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats,
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)


def check_for_augmented_data(data_dir):
    """Ask to use augmented data if `augmented_dogs.csv` exists in the data directory."""
    if "augmented_dogs.csv" in os.listdir(data_dir):
        print("Augmented data found, would you like to use it? y/n")
        print(">> ", end="")
        rep = str(input())
        return rep == "y"
    return False


def restore_checkpoint(model, checkpoint_dir, cuda=False, force=False, pretrain=False):
    """Restore model from checkpoint if it exists.

    Returns the model and the current epoch.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print(
            "Which epoch to load from? Choose in range [0, {}].".format(epoch),
            "Enter 0 to train from scratch.",
        )
        print(">> ", end="")
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(inp_epoch)
    )

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint["epoch"]
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats


def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def early_stopping(stats, curr_count_to_patience, global_min_loss):
    """Calculate new patience and validation loss.

    Increment curr_count_to_patience by one if new loss is not less than global_min_loss
    Otherwise, update global_min_loss with the current val loss, and reset curr_count_to_patience to 0

    Returns: new values of curr_count_to_patience and global_min_loss
    """
    # Implement early stopping
    idx = 0
    current_val_loss = stats[-1][idx]

    if not current_val_loss < global_min_loss:
        curr_count_to_patience += 1
    else:
        curr_count_to_patience = 0
        global_min_loss = current_val_loss
    
    return curr_count_to_patience, global_min_loss


def evaluate_epoch(
    axes,
    tr_loader,
    val_loader,
    te_loader,
    model,
    criterion,
    epoch,
    stats,
    device,
    include_test=False,
    update_plot=True,
):
    """Evaluate the `model` on the train and validation set."""

    def _get_metrics(loader):
        running_loss, running_r2 = [], []
        
        for batch in loader:
            with torch.no_grad():
                batch_inputs, batch_masks, batch_labels = \
                                 tuple(b.to(device) for b in batch)
                
                outputs = model(batch_inputs, batch_masks)
                
                running_loss.append(criterion(outputs, batch_labels).item())
                
                def _r2_score(outputs, labels):
                    labels_mean = torch.mean(labels)
                    ss_tot = torch.sum((labels - labels_mean) ** 2)
                    ss_res = torch.sum((labels - outputs) ** 2)
                    r2 = 1 - ss_res / ss_tot
                    return r2
        
                r2.append(_r2_score(outputs, batch_labels).item())

        loss = np.mean(running_loss)
        r2 = np.mean(running_r2)
        
        return loss, r2

    train_loss, train_r2 = _get_metrics(tr_loader)
    val_loss, val_r2 = _get_metrics(val_loader)

    stats_at_epoch = [
        val_loss,
        val_r2,
        train_loss,
        train_r2,
    ]
    if include_test:
        stats_at_epoch += list(_get_metrics(te_loader))

    stats.append(stats_at_epoch)
    utils.log_training(epoch, stats)
    if update_plot:
        utils.update_training_plot(axes, epoch, stats)


def train_epoch(data_loader, model, criterion, optimizer, device):
    """Train the `model` for one epoch of data from `data_loader`.

    Use `optimizer` to optimize the specified `criterion`
    """
    model.train()

    for step, batch in enumerate(data_loader):
        batch_inputs, batch_masks, batch_labels = \
                               tuple(b.to(device) for b in batch)
        optimizer.zero_grad()

        y_pred = model(batch_inputs, batch_masks)

        loss = criterion(y_pred.squeeze(), batch_labels.squeeze())

        # backward pass
        loss.backward()
        clip_grad_norm(model.parameters(), 2)
        optimizer.step()


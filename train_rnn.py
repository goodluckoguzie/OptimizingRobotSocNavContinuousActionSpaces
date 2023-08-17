import os
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import sys

# Constants
PROCESSED_DATA_PATH = "processed_data"
TRAIN_RATIO = 0.8
CHECKPOINT_DIR = 'ckpt'
RUNS_DIR = 'runs'
if not os.path.exists(PROCESSED_DATA_PATH):
    sys.exit("Error: 'processed_data' directory does not exist. Please ensure it is present before running the script.")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(RUNS_DIR):
    os.makedirs(RUNS_DIR)

class CombinedDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def load_and_combine_data(fpaths):
    all_inputs, all_targets = [], []

    for fpath in fpaths:
        episode_data = np.load(fpath)
        all_inputs.append(episode_data['input'])
        all_targets.append(episode_data['target'])

    return np.vstack(all_inputs), np.vstack(all_targets)

def create_datasets_and_loaders(data_path, batch_size=32):
    all_fpaths = sorted(glob.glob(os.path.join(data_path, 'processed_ep_*.npz')))
    train_size = int(TRAIN_RATIO * len(all_fpaths))
    train_fpaths = all_fpaths[:train_size]
    val_fpaths = all_fpaths[train_size:]
    train_inputs, train_targets = load_and_combine_data(train_fpaths)
    val_inputs, val_targets = load_and_combine_data(val_fpaths)
    train_inputs_tensor = torch.tensor(train_inputs, dtype=torch.float32)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32)
    val_inputs_tensor = torch.tensor(val_inputs, dtype=torch.float32)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.float32)
    train_dataset = CombinedDataset(train_inputs_tensor, train_targets_tensor)
    val_dataset = CombinedDataset(val_inputs_tensor, val_targets_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss, total_mae, total_rmse, total_samples = 0, 0, 0, len(loader.dataset)

    for batch_inputs, batch_targets in loader:
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_inputs)
        total_mae += torch.sum(torch.abs(outputs - batch_targets))
        total_rmse += torch.sum((outputs - batch_targets) ** 2)

    total_rmse = torch.sqrt(total_rmse / total_samples)
    return total_loss/total_samples, total_mae/total_samples, total_rmse

def validate_epoch(loader, model, criterion, device):
    model.eval()
    total_loss, total_mae, total_rmse, total_samples = 0, 0, 0, len(loader.dataset)
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item() * len(batch_inputs)
            total_mae += torch.sum(torch.abs(outputs - batch_targets))
            total_rmse += torch.sum((outputs - batch_targets) ** 2)

    total_rmse = torch.sqrt(total_rmse / total_samples)
    return total_loss/total_samples, total_mae/total_samples, total_rmse

if __name__ == "__main__":
    model_name = input("Enter the name to store the model and runs: ").strip()
    writer = SummaryWriter(os.path.join(RUNS_DIR, model_name))
    model_checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_name)
    model_folder = os.path.join(CHECKPOINT_DIR, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    train_loader, val_loader = create_datasets_and_loaders(PROCESSED_DATA_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ACTION_SIZE = 2
    OBS_SIZE = 51
    INPUT_SIZE = ACTION_SIZE + OBS_SIZE
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    OUTPUT_SIZE = OBS_SIZE
    EARLY_STOP = 20
    model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    best_avg_val_loss = float('inf')
    num_epochs = 10_000
    no_improvement_epochs = 0
    
    for epoch in range(num_epochs):
        train_start_time = time.time()
        avg_train_loss, avg_train_mae, avg_train_rmse = train_epoch(train_loader, model, criterion, optimizer, device)
        train_end_time = time.time()

        val_start_time = time.time()
        avg_val_loss, avg_val_mae, avg_val_rmse = validate_epoch(val_loader, model, criterion, device)
        val_end_time = time.time()

        scheduler.step(avg_val_loss)
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
        writer.add_scalar('MAE/Train', avg_train_mae, epoch)
        writer.add_scalar('MAE/Validation', avg_val_mae, epoch)
        writer.add_scalar('RMSE/Train', avg_train_rmse, epoch)
        writer.add_scalar('RMSE/Validation', avg_val_rmse, epoch)
      
        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train MAE: {avg_train_mae:.4f}, Val MAE: {avg_val_mae:.4f}, Train RMSE: {avg_train_rmse:.4f},  Val RMSE: {avg_val_rmse:.4f}")
            print(f"Training time for epoch {epoch}: {train_end_time - train_start_time:.2f} seconds")
            print(f"Validation time for epoch {epoch}: {val_end_time - val_start_time:.2f} seconds")


        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            no_improvement_epochs = 0
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            # torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f'{model_name}_{epoch}.pth.tar'))
            torch.save(checkpoint, os.path.join(model_folder, f'{model_name}_{epoch}.pth.tar'))

        else:
            no_improvement_epochs += 1
        if no_improvement_epochs >=EARLY_STOP:
            print("Early stopping due to no improvement in validation loss for X consecutive epochs.")
            break
    writer.close()

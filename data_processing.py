import numpy as np
import sys
import os
import glob
import torch
from hparams import RobotFrame_Continuous_Datasets_Timestep_1 as data

# Constants
WINDOW_SIZE = 16
PAD_VALUE = 0
PROCESSED_DATA_PATH = "processed_data"
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

# Data Normalization Constants
OBS_MIN = -14.142136
OBS_MAX = 14.142136
ACTION_MIN = -1.
ACTION_MAX = 1.

def normalize_observation(obs):
    return (obs - OBS_MIN) / (OBS_MAX - OBS_MIN)

def normalize_action(action):
    return (action - ACTION_MIN) / (ACTION_MAX - ACTION_MIN)

def apply_padding(obs, actions, window_size):
    pad_size_front = window_size - 1
    pad_size_end = window_size // 2
    obs_padding_front = PAD_VALUE * torch.ones([pad_size_front, obs.size(1)])
    obs_padding_end = PAD_VALUE * torch.ones([pad_size_end, obs.size(1)])
    obs = torch.cat([obs_padding_front, obs, obs_padding_end], dim=0)
    
    actions_padding_front = PAD_VALUE * torch.ones([pad_size_front, actions.size(1)])
    actions_padding_end = PAD_VALUE * torch.ones([pad_size_end, actions.size(1)])
    actions = torch.cat([actions_padding_front, actions, actions_padding_end], dim=0)
    
    return obs, actions

def process_episode_data(data_path):
    print("loading the dataset.............")
    all_fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_ep_*.npz')))

    print("applying padding to  the dataset. and Sliding window to create input-target pairs...........")
    for idx, fpath in enumerate(all_fpaths):
        episode_data = np.load(fpath)
        obs = normalize_observation(torch.from_numpy(episode_data['obs']))
        actions = normalize_action(torch.from_numpy(episode_data['action']))
        
        # Apply padding
        obs, actions = apply_padding(obs, actions, WINDOW_SIZE)

        # Sliding window to create input-target pairs
        input_data, target_data = [], []
        for i in range(len(obs) - WINDOW_SIZE):
            input_data.append(torch.cat((obs[i:i+WINDOW_SIZE], actions[i:i+WINDOW_SIZE]), dim=-1).numpy())
            target_data.append(obs[i+WINDOW_SIZE].numpy())

        # Save processed data in .npz format
        npz_filename = os.path.join(PROCESSED_DATA_PATH, f"processed_ep_{idx}.npz")
        np.savez(npz_filename, input=np.stack(input_data), target=np.stack(target_data), window_size=WINDOW_SIZE)
    print("Done.....Saving.......>.")
    print("..............Done...........")

if __name__ == "__main__":
    DATA_DIR = data.data_dir

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        sys.exit(f"Error: '{DATA_DIR}' directory does not exist. Please ensure it is present before running the script.")

    process_episode_data(DATA_DIR)

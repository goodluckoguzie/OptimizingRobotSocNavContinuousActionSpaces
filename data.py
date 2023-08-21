
import os, sys, glob
import numpy as np
import torch
from collections import defaultdict




class GameSceneDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, training=True, test_ratio=0.01):
        self.fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_[0-9][0-9][0-9]_*.npz')))
        np.random.seed(0)
        indices = np.arange(0, len(self.fpaths))
        n_trainset = int(len(indices)*(1.0-test_ratio))
        self.train_indices = indices[:n_trainset]
        self.test_indices = indices[n_trainset:]
        # self.train_indices = np.random.choice(indices, int(len(indices)*(1.0-test_ratio)), replace=False)
        # self.test_indices = np.delete(indices, self.train_indices)
        self.indices = self.train_indices if training else self.test_indices
        # import pdb; pdb.set_trace()

    def __getitem__(self, idx):
        npz = np.load(self.fpaths[self.indices[idx]])
        obs = npz['obs']
        # obs = transform(obs)
        # obs = obs.permute(2, 0, 1) # (N, C, H, W)
        return obs

    def __len__(self):
        return len(self.indices)

def collate_fn(data):
    obs, actions = zip(*data)
    obs, actions = np.array(obs), np.array(actions)

    _,_, seq_len, C = obs.shape

    obs = obs.reshape([-1, C]) # (B*N_seq*seq_len, H, W, C)
    actions = actions.reshape([-1, actions.shape[-1]]) # (B*N_seq*seq_len, H, W, C)

    obs_lst, actions_lst = [], []
    
    for ob, action in zip(obs, actions):
        obs_lst.append(torch.from_numpy(ob))
        actions_lst.append(torch.from_numpy(action))

    obs = torch.stack(obs_lst, dim=0) # (B*N_seq*seq_len, C, H, W)
    actions = torch.stack(actions_lst, dim=0) # (B*N_seq*seq_len, n_actions)

    return obs, actions
    

class GameEpisodeDatasetNonPrePadded(torch.utils.data.Dataset):

    def __init__(self, data_path, seq_len=20, seq_mode=True, training=True, test_ratio=0.2, episode_length=None):
        self.training = training
        self.episode_length = episode_length
        
        # Load all the file paths
        all_fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_ep_*.npz')))
        
        # Filter out episodes with zero lengths
        self.fpaths = [f for f in all_fpaths if np.load(f)['obs'].shape[0] > 0]
        
        np.random.seed(0)

        indices = np.arange(0, len(self.fpaths))
        n_trainset = int(len(indices) * (1.0 - test_ratio))
        
        self.train_indices = indices[:n_trainset]
        self.test_indices = indices[n_trainset:]
        self.indices = self.train_indices if training else self.test_indices
        self.seq_len = seq_len
        self.seq_mode = seq_mode

    def __getitem__(self, idx):
        def pad_tensor(t, episode_length=self.episode_length, window_length=self.seq_len-1, pad_function=torch.zeros):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            pad_size = episode_length - t.size(0) + window_length
            # Add window lenght - 1 infront of the number of obersavtion
            begin_pad       = pad_function([window_length-1, t.size(1)]).to(device)
            # pad the environment with lenght of the episode subtracted from  the total episode length
            episode_end_pad = pad_function([pad_size,      t.size(1)]).to(device)
            # print("paaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",episode_end_pad.shape)

            return torch.cat([begin_pad,t.to(device),episode_end_pad], dim=0)

        npz = np.load(self.fpaths[self.indices[idx]])
        obs = npz['obs'] # (T, H, W, C) np array
        obs = torch.from_numpy(obs) 

        obs = pad_tensor(obs ,window_length=(self.seq_len-1)).cpu().detach().numpy()
        actions = npz['action'] # (, n_actions) np array

        actions = torch.from_numpy(actions) 
        actions = pad_tensor(actions, window_length=(self.seq_len-1)).cpu().detach().numpy()

        k,h = actions.shape
        T, C = obs.shape
        # T, H, W, C = obs.shape
        n_seq = T // self.seq_len
        end_seq = n_seq * self.seq_len # T' = end of sequence

        # print("end_seqend_seqend_seqend_seqend_seqend_seqend_seq",end_seq)          
        obs =     obs[:end_seq].reshape([-1, self.seq_len, C]) # (N_seq, seq_len, H, W, C)
        actions = actions[:end_seq].reshape([-1, self.seq_len, actions.shape[-1]]) # 
        return obs, actions

    def __len__(self):
        return len(self.indices)





class CustomEpisodeLoader:
    def __init__(self, data_path, batch_size, mode='train'):
        assert mode in ['train', 'val'], "Mode should be 'train' or 'val'"

        self.data_path = data_path
        self.batch_size = batch_size
        self.mode = mode
        self._current_idx = 0

        # Categorize episodes by their length
        all_episodes = []

        for filename in os.listdir(data_path):
            if filename.startswith('rollout_ep_') and filename.endswith('.npz'):
                filepath = os.path.join(data_path, filename)
                with np.load(filepath) as data:
                    obs = data['obs']
                    actions = data['action']

                    # Check for episode length and skip if less than 2
                    if obs.shape[0] < 2:
                        continue

                    all_episodes.append((obs, actions))

        total_episodes = len(all_episodes)
        train_cutoff = int(0.8 * total_episodes)

        if mode == 'train':
            all_episodes = all_episodes[:train_cutoff]
        else:
            all_episodes = all_episodes[train_cutoff:]

        self.episodes = defaultdict(list)
        for obs, actions in all_episodes:
            episode_length = obs.shape[0]
            self.episodes[episode_length].append((obs, actions))

    def batch_generator(self):
        for length, eps in self.episodes.items():
            num_eps = len(eps)
            for i in range(0, num_eps, self.batch_size):
                batch = eps[i: i + self.batch_size]
                
                # Convert to tensors here
                obs_batch = torch.tensor([ep[0] for ep in batch], dtype=torch.float32)
                actions_batch = torch.tensor([ep[1] for ep in batch], dtype=torch.float32)

                yield obs_batch, actions_batch

    def __iter__(self):
        self._current_idx = 0
        self._all_batches = list(self.batch_generator())  # Pre-compute all batches
        return self

    def __next__(self):
        if self._current_idx < len(self._all_batches):
            batch = self._all_batches[self._current_idx]
            self._current_idx += 1
            return batch
        raise StopIteration




########################################################################################################################################
import numpy as np
import torch
import os
import glob

class rewardGameEpisodeDatasetNonPrePadded(torch.utils.data.Dataset):

    def __init__(self, data_path, seq_len=20, seq_mode=True, training=True, test_ratio=0.2, episode_length=None):
        self.training = training
        self.episode_length = episode_length
        
        # Load all the file paths
        all_fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_ep_*.npz')))
        
        # Filter out episodes with zero lengths
        self.fpaths = [f for f in all_fpaths if np.load(f)['obs'].shape[0] > 0]
        
        np.random.seed(0)

        indices = np.arange(0, len(self.fpaths))
        n_trainset = int(len(indices) * (1.0 - test_ratio))
        
        self.train_indices = indices[:n_trainset]
        self.test_indices = indices[n_trainset:]
        self.indices = self.train_indices if training else self.test_indices
        self.seq_len = seq_len
        self.seq_mode = seq_mode

    def __getitem__(self, idx):

        def pad_tensor(t, episode_length=self.episode_length, window_length=self.seq_len-1, pad_function=torch.zeros):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pad_size = episode_length - t.size(0) + window_length
            episode_end_pad = pad_function([pad_size, t.size(1)]).to(device)
            return torch.cat([t.to(device), episode_end_pad], dim=0)

        npz = np.load(self.fpaths[self.indices[idx]])

        obs = npz['obs']
        obs = torch.from_numpy(obs) 
        obs = pad_tensor(obs, window_length=(self.seq_len-1)).cpu().detach().numpy()

        actions = npz['action']
        actions = torch.from_numpy(actions)
        actions = pad_tensor(actions, window_length=(self.seq_len-1)).cpu().detach().numpy()

        rewards = npz['reward']
        rewards = torch.from_numpy(rewards)
        rewards = pad_tensor(rewards, window_length=(self.seq_len-1)).cpu().detach().numpy()

        k, h = actions.shape
        T, C = obs.shape

        n_seq = T // self.seq_len
        end_seq = n_seq * self.seq_len

        obs = obs[:end_seq].reshape([-1, self.seq_len, C])
        actions = actions[:end_seq].reshape([-1, self.seq_len, actions.shape[-1]])
        rewards = rewards[:end_seq].reshape([-1, self.seq_len, rewards.shape[-1]])

        return obs, actions, rewards

    def __len__(self):
        return len(self.indices)



def rewardcollate_fn(data):
    # obs (B, N_seq, seq_len, H, W, C), actions (B, N_seq, seq_len, n_actions), rewards (B, N_seq, seq_len, 1)
    obs, actions, rewards = zip(*data)
    obs, actions, rewards = np.array(obs), np.array(actions), np.array(rewards)

    _, _, seq_len, C = obs.shape

    obs = obs.reshape([-1, C]) # (B*N_seq*seq_len, H, W, C)
    actions = actions.reshape([-1, actions.shape[-1]]) # (B*N_seq*seq_len, n_actions)
    rewards = rewards.reshape([-1, rewards.shape[-1]]) # (B*N_seq*seq_len, 1)

    obs_lst, actions_lst, rewards_lst = [], [], []
    
    for ob, action, reward in zip(obs, actions, rewards):
        obs_lst.append(torch.from_numpy(ob))
        actions_lst.append(torch.from_numpy(action))
        rewards_lst.append(torch.from_numpy(reward))

    obs = torch.stack(obs_lst, dim=0) # (B*N_seq*seq_len, C, H, W)
    actions = torch.stack(actions_lst, dim=0) # (B*N_seq*seq_len, n_actions)
    rewards = torch.stack(rewards_lst, dim=0) # (B*N_seq*seq_len, 1)

    return obs, actions, rewards
#######################################################################################################################################
# Global min-max for observation
OBS_MIN = -14.142136
OBS_MAX = 14.142136
# Global min-max for action
ACTION_MIN = -1.
ACTION_MAX = 1.

def normalised_collate_fn(data):
    obs, actions = zip(*data)
    obs, actions = np.array(obs), np.array(actions)

    _,_, seq_len, C = obs.shape

    obs = obs.reshape([-1, C]) # (B*N_seq*seq_len, H, W, C)
    actions = actions.reshape([-1, actions.shape[-1]]) # (B*N_seq*seq_len, H, W, C)

    obs_lst, actions_lst = [], []
    
    for ob, action in zip(obs, actions):
        obs_lst.append(torch.from_numpy(ob))
        actions_lst.append(torch.from_numpy(action))

    obs = torch.stack(obs_lst, dim=0) # (B*N_seq*seq_len, C, H, W)
    actions = torch.stack(actions_lst, dim=0) # (B*N_seq*seq_len, n_actions)

    return obs, actions
    
def normalize(value, min_value, max_value):
    """Normalizes the input value using the min and max values."""
    return (value - min_value) / (max_value - min_value)

class NormalisedGameEpisodeDatasetNonPrePadded(torch.utils.data.Dataset):

    def __init__(self, data_path, seq_len=20, seq_mode=True, training=True, test_ratio=0.2, episode_length=None):
        self.training = training
        self.episode_length = episode_length
        
        # Load all the file paths
        all_fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_ep_*.npz')))
        
        # Filter out episodes with zero lengths
        self.fpaths = [f for f in all_fpaths if np.load(f)['obs'].shape[0] > 0]
        
        np.random.seed(0)

        indices = np.arange(0, len(self.fpaths))
        n_trainset = int(len(indices) * (1.0 - test_ratio))
        
        self.train_indices = indices[:n_trainset]
        self.test_indices = indices[n_trainset:]
        self.indices = self.train_indices if training else self.test_indices
        self.seq_len = seq_len
        self.seq_mode = seq_mode

    def __getitem__(self, idx):
        # This function pads a tensor 't' based on the episode length and window length
        def pad_tensor(t, episode_length=self.episode_length, window_length=self.seq_len-1, pad_function=torch.zeros):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pad_size = episode_length - t.size(0) + window_length
            episode_end_pad = pad_function([pad_size, t.size(1)]).to(device)
            return torch.cat([t.to(device), episode_end_pad], dim=0)

        # Load the episode data from file
        npz = np.load(self.fpaths[self.indices[idx]])
        obs = npz['obs'] # Extract observations
        obs = torch.from_numpy(obs)
        obs = pad_tensor(obs ,window_length=(self.seq_len-1)).cpu().detach().numpy()
        
        actions = npz['action'] # Extract actions
        actions = torch.from_numpy(actions)
        actions = pad_tensor(actions, window_length=(self.seq_len-1)).cpu().detach().numpy()

        # Normalize observations and actions
        obs = normalize(obs, OBS_MIN, OBS_MAX)
        actions = normalize(actions, ACTION_MIN, ACTION_MAX)

        # Reshape the observation and action data
        T, C = obs.shape
        n_seq = T // self.seq_len
        end_seq = n_seq * self.seq_len
        obs = obs[:end_seq].reshape([-1, self.seq_len, C])
        actions = actions[:end_seq].reshape([-1, self.seq_len, actions.shape[-1]])

        return obs, actions

    def __len__(self):
        return len(self.indices)
################################################################################################################################################   new ####################################################################

# import torch
# import numpy as np
# import os
# import glob

# # Constants
# PAD_VALUE = 5

# def create_mask(tensor, pad_value=PAD_VALUE):
#     return (tensor != pad_value).float()

# class GameEpisodeDatasetNonPrePadded(torch.utils.data.Dataset):
    
#     def __init__(self, data_path, seq_len=20, seq_mode=True, training=True, test_ratio=0.2, episode_length=None, random_seed=0):
#         self.training = training
#         self.episode_length = episode_length
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # Load all the file paths
#         all_fpaths = sorted(glob.glob(os.path.join(data_path, 'rollout_ep_*.npz')))
        
#         # Filter out episodes with zero lengths
#         filtered_fpaths = [f for f in all_fpaths if np.load(f)['obs'].shape[0] > 0]
#         num_filtered = len(all_fpaths) - len(filtered_fpaths)
#         if num_filtered:
#             print(f"Filtered out {num_filtered} episodes with zero length.")
#         self.fpaths = filtered_fpaths
        
#         np.random.seed(random_seed)

#         indices = np.arange(0, len(self.fpaths))
#         n_trainset = int(len(indices) * (1.0 - test_ratio))
        
#         self.train_indices = indices[:n_trainset]
#         self.test_indices = indices[n_trainset:]
#         self.indices = self.train_indices if training else self.test_indices
#         self.seq_len = seq_len
#         self.seq_mode = seq_mode

#     def __getitem__(self, idx):
#         def pad_tensor(t, episode_length=self.episode_length, window_length=self.seq_len, pad_value=5):
#             last_value = t[-1:]  # Capture the last value
#             t = t[:-1]  # Remove the last value
#             pad_size = episode_length - t.size(0) - 1
#             episode_end_pad = pad_value * torch.ones([pad_size, t.size(1)], device=self.device)
#             padded_tensor = torch.cat([t, episode_end_pad], dim=0)
#             return torch.cat([padded_tensor, last_value], dim=0)  # Add the last value back to the padded tensor


#         npz = np.load(self.fpaths[self.indices[idx]])
#         obs = torch.from_numpy(npz['obs']).to(self.device)
        
#         mask = create_mask(obs)

#         mask = pad_tensor(mask, window_length=(self.seq_len)).float()
        
#         obs = pad_tensor(obs, window_length=(self.seq_len))

#         actions = torch.from_numpy(npz['action']).to(self.device)
#         actions = pad_tensor(actions, window_length=(self.seq_len))

#         k, h = actions.shape
#         T, C = obs.shape

#         n_seq = T // self.seq_len
#         end_seq = n_seq * self.seq_len

#         # Squeeze out the unnecessary dimension
#         obs = obs.squeeze(0)
#         actions = actions.squeeze(0)
#         mask = mask.squeeze(0)

#         obs = obs[:end_seq].reshape([-1, self.seq_len, C])
#         actions = actions[:end_seq].reshape([-1, self.seq_len, actions.shape[-1]])
#         mask = mask[:end_seq].reshape([-1, self.seq_len, mask.shape[-1]])

#         # print("Observations shape:", obs.shape)
#         # print("Actions shape:", actions.shape)
#         # print("Mask shape:", mask.shape)

#         # Assertion to ensure shapes match
#         assert obs.shape[1] == actions.shape[1] == mask.shape[1], "Mismatch in sequence lengths after processing."

#         return obs, actions, mask  # return the mask alongside obs and actions

#     def __len__(self):
#         return len(self.indices)

# def collate_fn(data):
#     obs, actions, masks = zip(*data)  # Unzip the data into their respective categories
#     obs = torch.stack(obs, dim=0).squeeze(1)       # Stack tensors along a new dimension and squeeze
#     actions = torch.stack(actions, dim=0).squeeze(1)
#     masks = torch.stack(masks, dim=0).squeeze(1)

#     return obs, actions, masks  # Return the data including masks


##############################################################################################################################
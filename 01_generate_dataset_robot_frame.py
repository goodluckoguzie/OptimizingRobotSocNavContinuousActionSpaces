import numpy as np
import os, sys, glob
import gym
from hparams import HyperParams as hp
from hparams import RobotFrame_Continuous_Datasets_Timestep_1 as data
import sys
sys.path.append('./gsoc22-socnavenv')
import random
import socnavenv
from socnavenv.wrappers import WorldFrameObservations
import os
import torch

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# def discrete_to_continuous_action(action:int):
#     """
#     Function to return a continuous space action for a given discrete action
#     """
#     # 0.25, 0.5, 0.75 denote slower movements
#     # -0.25, -0.5, -0.75 denote backwards movements
#     move_dict = {0: 0.25, 1: 0.5, 2: 0.75, 3: 1, 4: 0, 5: -0.25, 6: -0.5, 7: -0.75}

#     # 0.25, 0.5, 0.75 denote smaller rotations
#     # -0.25, -0.5, -0.75 denote larger rotations
#     turn_dict = {0: 0.25, 1: 0.5, 2: 0.75, 3: 1, 4: 0, 5: -0.25, 6: -0.5, 7: -0.75}

#     move = move_dict[action // 8] 
#     turn = turn_dict[action % 8] 

#     return np.array([move, turn], dtype=np.float32)
def discrete_to_continuous_action(action:int):
    """
    Function to return a continuous space action for a given discrete action
    """
    # Adjust the possible action values to be either -0.5, 0, 0.5 or 1
    move_dict = {0: -0.5, 1: 0, 2: 0.5, 3: 1}
    turn_dict = {0: -0.5, 1: 0, 2: 0.5, 3: 1}

    move = move_dict[action // 4] 
    turn = turn_dict[action % 4] 

    return np.array([move, turn], dtype=np.float32)

def preprocess_observation(obs):
    """
    To convert dict observation to numpy observation
    """
    assert(type(obs) == dict)
    obs2 = np.array(obs["goal"][-2:], dtype=np.float32)
    humans = obs["humans"].flatten()
    for i in range(int(round(humans.shape[0]/(6+7)))):
        index = i*(6+7)
        obs2 = np.concatenate((obs2, humans[index+6:index+6+7]) )

    # laptops = obs["laptops"].flatten()
    # for i in range(int(round(laptops.shape[0]/(6+7)))):
    #     index = i*(6+7)
    #     obs2 = np.concatenate((obs2, laptops[index+6:index+6+7]) )

    tables = obs["tables"].flatten()
    for i in range(int(round(tables.shape[0]/(6+7)))):
        index = i*(6+7)
        obs2 = np.concatenate((obs2, tables[index+6:index+6+7]) )

    plants = obs["plants"].flatten()
    for i in range(int(round(plants.shape[0]/(6+7)))):
        index = i*(6+7)
        obs2 = np.concatenate((obs2, plants[index+6:index+6+7]) )
    return torch.from_numpy(obs2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def pad_tensor(t, episode_length, window_length=16, pad_function=torch.zeros):
    pad_size = episode_length - t.size(0) + window_length
    episode_end_pad = pad_function([pad_size,      t.size(1)]).to(device)
    return torch.cat([t.to(device),episode_end_pad], dim=0)


def get_space_bounds(space):
    if isinstance(space, gym.spaces.Box):
        return space.low, space.high
    elif isinstance(space, gym.spaces.Discrete):
        return 0, space.n
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return np.zeros_like(space.nvec), space.nvec
    elif isinstance(space, gym.spaces.MultiBinary):
        return np.zeros_like(space.n), np.ones_like(space.n)
    elif isinstance(space, gym.spaces.Dict):
        min_dict = {}
        max_dict = {}
        for key, item in space.spaces.items():
            min_val, max_val = get_space_bounds(item)
            min_dict[key] = min_val
            max_dict[key] = max_val
        return min_dict, max_dict
    else:
        print(f'Type of space: {type(space)}')  # This will print the type of the space if it's not recognized
        raise NotImplementedError('This space type is not yet supported for getting bounds')


def rollout():
    time_steps = data.time_steps
    env = gym.make("SocNavEnv-v1")
    env.configure('./configs/env_timestep_1.yaml')
    env.set_padded_observations(True)

    # Get observation and action spaces
    observation_space = env.observation_space
    action_space = env.action_space

    # Get min and max of observation space
    min_observation, max_observation = get_space_bounds(observation_space)
    print('Min observation:', min_observation)
    print('Max observation:', max_observation)

    # Get min and max of action space
    min_action, max_action = get_space_bounds(action_space)
    print('Min action:', min_action)
    print('Max action:', max_action)



    # Extracting global min and max from observation dictionaries, skipping empty arrays
    global_min = min([np.min(min_observation[key]) for key in min_observation if min_observation[key].size > 0])
    global_max = max([np.max(max_observation[key]) for key in max_observation if max_observation[key].size > 0])

    print("Global Min:", global_min)
    print("Global Max:", global_max)

    max_ep = 10000
    feat_dir = data.data_dir
    os.makedirs(feat_dir, exist_ok=True)
    env.seed(1) # deterministic for demonstration

    for ep in range(max_ep):
        obs_lst, action_lst, reward_lst, next_obs_lst, done_lst = [], [], [], [], []
        obs = env.reset()
        obs = preprocess_observation(obs)   

        done = False
        t = 0

        for t in range(time_steps+10): 
            # env.render()      

            # action_ = np.random.randint(0, 64)  # Updated from 4 to 64, to match the new action space
            # action = discrete_to_continuous_action(action_)
            # action = np.round(action, decimals=2)
            # print(action)
            action_ = np.random.randint(0, 16)  # Updated from 4 to 16, to match the new action space
            # action_ = 9  # Choose an initial action for demonstration purposes
            action = discrete_to_continuous_action(action_)
            action = np.round(action, decimals=2)
            # print(action)


            next_obs, reward, done, _ = env.step(action)
            next_obs = preprocess_observation(next_obs)
            action = torch.from_numpy(action)

            np.savez(
                os.path.join(feat_dir, 'rollout_{:03d}_{:04d}'.format(ep,t)),
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )

            obs_lst.append(obs)
            action_lst.append(action)
            reward_lst.append(reward)
            next_obs_lst.append(next_obs)
            done_lst.append(done)
            obs = next_obs
            if done:
                print("Episode [{}/{}] finished after {} timesteps".format(ep + 1, max_ep, t), flush=True)
                obs = env.reset()
                obs_lst = torch.stack(obs_lst, dim=0).squeeze(1)
                next_obs_lst = torch.stack(next_obs_lst, dim=0).squeeze(1)
                done_lst = [int(d) for d in done_lst]
                done_lst = torch.tensor(done_lst).unsqueeze(-1)
                action_lst = torch.stack(action_lst, dim=0).squeeze(1)
                reward_lst = torch.tensor(reward_lst).unsqueeze(-1)
                break

        np.savez(
            os.path.join(feat_dir, 'rollout_ep_{:03d}'.format(ep)),
            obs=np.stack(obs_lst, axis=0), # (T, C, H, W)
            action=np.stack(action_lst, axis=0), # (T, a)
            reward=np.stack(reward_lst, axis=0), # (T, 1)
            next_obs=np.stack(next_obs_lst, axis=0), # (T, C, H, W)
            done=np.stack(done_lst, axis=0), # (T, 1)
        )

if __name__ == '__main__':
    np.random.seed(123)
    rollout()

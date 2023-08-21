import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import numpy as np
from hparams import VAEHyperParams as hp
from data import *
import os, sys
import gym
import random
import cv2
from matplotlib.pyplot import axis
sys.path.append('./gsoc22-socnavenv')
import socnavenv
from collections import deque


from socnavenv.envs.utils.human import Human
from socnavenv.envs.utils.laptop import Laptop
from socnavenv.envs.utils.plant import Plant
from socnavenv.envs.utils.table import Table
from socnavenv.envs.utils.robot import Robot
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = 53


RESOLUTION_VIEW = None
window_initialised = False
RESOLUTION_VIEW = 1000
MAP_X = random.randint(10, 10)
MAP_Y = random.randint(10, 10)
RESOLUTION_X = int(1500 * MAP_X/(MAP_X + MAP_Y))
RESOLUTION_Y = int(1500 * MAP_Y/(MAP_X + MAP_Y))
PIXEL_TO_WORLD_X = RESOLUTION_X / MAP_X
PIXEL_TO_WORLD_Y = RESOLUTION_Y / MAP_Y
GOAL_RADIUS = 0.5


def w2px(x, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given x-coordinate in world frame, to get the x-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * (x + (MAP_SIZE / 2)))


def w2py(y, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given y-coordinate in world frame, to get the y-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * ((MAP_SIZE / 2) - y))


def get_observation_from_dataset(dataset, idx):
    sample = dataset[idx,:]
    return transform_processed_observation_into_raw(sample)

def transform_processed_observation_into_raw(sample):

    ind_pos = [0,1]
    goal_obs = [sample[i] for i in ind_pos]
    goal_obs = np.array(goal_obs)
    # print(goal_obs)
    humans = []
    for human_num in range(2, sample.size()[0],7):
        humans.append(sample[human_num:human_num + 3])


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNN(nn.Module):
    def __init__(self, n_latents, n_actions, n_hiddens, n_layers, dropout_rate=0.5):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(n_latents + n_actions, n_hiddens, num_layers=n_layers, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Deepening the network with more layers and adding residual connections.
        self.fc1 = nn.Linear(n_hiddens, n_hiddens) 
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)  
        self.fc3 = nn.Linear(n_hiddens, n_hiddens) 
        self.fc4 = nn.Linear(n_hiddens, n_hiddens)
        self.fc5 = nn.Linear(n_hiddens, n_hiddens)
        self.fc6 = nn.Linear(n_hiddens, n_hiddens)
        self.fc7 = nn.Linear(n_hiddens, n_hiddens)
        
        self.fc_out = nn.Linear(n_hiddens, n_latents)

        self.activation = nn.ReLU()

    def forward(self, states):
        h, _ = self.rnn(states)
        identity = h
        
        # Implementing the deeper network with residual connections.
        y = self.activation(self.fc1(h))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc2(y))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc3(y))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc4(y))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc5(y))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc6(y))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc7(y))
        y = self.dropout(y)
        y += identity

        y = self.fc_out(y)
        return y, None, None

    def infer(self, states, hidden):
        h, next_hidden = self.rnn(states, hidden)
        identity = h
        
        # Implementing the deeper network with residual connections for inference.
        y = self.activation(self.fc1(h))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc2(y))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc3(y))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc4(y))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc5(y))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc6(y))
        y = self.dropout(y)
        y += identity

        y = self.activation(self.fc7(y))
        y = self.dropout(y)
        y += identity

        y = self.fc_out(y)
        return y, None, None, next_hidden




# Data Normalization Constants
OBS_MIN = -14.142136
OBS_MAX = 14.142136
ACTION_MIN = -1.
ACTION_MAX = 1.

def normalize_observation(obs):
    normalized_0_1 = (obs - OBS_MIN) / (OBS_MAX - OBS_MIN)
    return 2 * normalized_0_1 - 1

def normalize_action(action):
    normalized_0_1 = (action - ACTION_MIN) / (ACTION_MAX - ACTION_MIN)
    return 2 * normalized_0_1 - 1


def denormalize_observation(normalized_obs):
    denormalized_0_1 = (normalized_obs + 1) / 2
    return denormalized_0_1 * (OBS_MAX - OBS_MIN) + OBS_MIN

def denormalize_action(normalized_action):
    denormalized_0_1 = (normalized_action + 1) / 2
    return denormalized_0_1 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN


# Get the subfolder name from the user
model_run_name = input("Please enter the subfolder name of the trained model: ")

# Load the trained model
ckpt_dir = os.path.join('ckpt',  model_run_name)

n_latents = 51
n_actions = 2
hiddens = 256 #256
n_layers = 3 #1 #2
K=1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

rnn = RNN(n_latents, n_actions, hiddens, n_layers).to(device)

rnn_ckpt_path = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth.tar')))[-1]

# rnn_state = torch.load(rnn_ckpt_path, map_location={'cuda:0': str(device)})
rnn_state = torch.load(rnn_ckpt_path, map_location=device)

rnn.load_state_dict(rnn_state['model'])
rnn.eval()
print(f'Loaded RNN checkpoint {rnn_ckpt_path}')     

# cv2.namedWindow("input", cv2.WINDOW_NORMAL) 
# cv2.resizeWindow("input", int(socnavenv.RESOLUTION_VIEW*0.5), int(socnavenv.RESOLUTION_VIEW*0.5))
# cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
# cv2.resizeWindow("output", int(socnavenv.RESOLUTION_VIEW*0.5), int(socnavenv.RESOLUTION_VIEW*0.5))
RESOLUTION_VIEW = 1000
MAP_X = random.randint(10, 10)
MAP_Y = random.randint(10, 10)

RESOLUTION_X = int(1500 * MAP_X/(MAP_X + MAP_Y))
RESOLUTION_Y = int(1500 * MAP_Y/(MAP_X + MAP_Y))
PIXEL_TO_WORLD_X = RESOLUTION_X / MAP_X
PIXEL_TO_WORLD_Y = RESOLUTION_Y / MAP_Y
GOAL_RADIUS = 0.5


def w2px(x, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given x-coordinate in world frame, to get the x-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * (x + (MAP_SIZE / 2)))


def w2py(y, PIXEL_TO_WORLD, MAP_SIZE):
    """
    Given y-coordinate in world frame, to get the y-coordinate in the image frame
    """
    return int(PIXEL_TO_WORLD * ((MAP_SIZE / 2) - y))


def get_observation_from_dataset(dataset, idx):
    sample = dataset[idx,:]
    return transform_processed_observation_into_raw(sample)

def transform_processed_observation_into_raw(sample):

    ind_pos = [0,1]
    goal_obs = [sample[i] for i in ind_pos]
    goal_obs = np.array(goal_obs)
    # print(goal_obs)
    humans = []
    for human_num in range(2, sample.size()[0],7):
        humans.append(sample[human_num:human_num + 4])

    return goal_obs, humans        

rollout_dir = 'Data/'
if not os.path.exists(rollout_dir):
    os.makedirs(rollout_dir)


def get_observation_from_dataset(dataset, idx):
    sample = dataset[idx,:]
    return transform_processed_observation_into_raw(sample)

def transform_processed_observation_into_raw(sample):
    # print("sample", sample)

    goal_obs = sample[0:2]  # First two values are for the goal

    humans = [sample[i:i + 7] for i in range(2, 2 + 5 * 7, 7)]  # 5 humans each with 7 values
    # print("humanshumanshumans", humans)

    # Assuming that the order is goals, humans, laptops, tables, plants.
    # laptops_start = 2 + 5 * 7
    # laptops = sample[laptops_start:laptops_start + 7]

    tables_start = 2 + 5 * 7
    # print("tables_start", tables_start)
    tables = sample[tables_start:tables_start + 7]

    # tables_start = laptops_start + 7
    # tables = sample[tables_start:tables_start + 7]

    plants_start = tables_start + 7
    # print("plants_start", plants_start)

    plants = sample[plants_start:plants_start + 7]
    # print("plantsplants", plants)

    # return goal_obs, humans, laptops, tables, plants
    return goal_obs, humans, tables, plants


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

def draw_observation(image, humans_obs, tables_obs, plants_obs, goal_obs):
    Human_list = []
    for obs in humans_obs:
        Human_list.append(Human(id=1, x=obs[0], y=obs[1], theta=obs[2], width=0.72 ))
    for human in Human_list:
        human.draw(image, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_X, MAP_Y)

    Table_list = []
    if isinstance(tables_obs, torch.Tensor):
        tables_obs = [tables_obs]
    for obs in tables_obs:
        Table_list.append(Table(id=1, x=obs[0], y=obs[1], theta=obs[2], width=1.5, length=3.0 ))
    for table in Table_list:
        table.draw(image, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_X, MAP_Y)
        
    Plant_list = []
    if isinstance(plants_obs, torch.Tensor):
        plants_obs = [plants_obs]               
    for obs in plants_obs:
        Plant_list.append(Plant(id=1, x=obs[0], y=obs[1], radius=0.4 ))
    for plant in Plant_list:
        plant.draw(image, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y, MAP_X, MAP_Y)
    cv2.circle(image, (w2px(goal_obs[0], PIXEL_TO_WORLD_X, MAP_X), w2py(goal_obs[1], PIXEL_TO_WORLD_Y, MAP_Y)), int(w2px(0 + GOAL_RADIUS,PIXEL_TO_WORLD_X, MAP_X) - w2px(0, PIXEL_TO_WORLD_X, MAP_X)), (0, 255, 0), 2)
    #draw the robot
    robot = Robot(id=1, x=0, y=0, theta=0, radius=0.3  )
    robot.draw(image, PIXEL_TO_WORLD_X, PIXEL_TO_WORLD_Y,MAP_X,MAP_Y)
    input_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return input_grey


    
class Rollout():
    def __init__(self, data_dic, dir_name,mode, num_episodes_to_record):
        super().__init__()
        self.data_dic = data_dic
        self.dir_name = dir_name
        self.mode = mode
        self.num_episodes_to_record = num_episodes_to_record
        
    def make_rollout(self):
        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        env.seed(1)

        s = 0
        while s < self.num_episodes_to_record:
            next_hidden = None 
            obs_queue = deque(maxlen=K + 1)
            action_queue = deque(maxlen=K + 1)
            obs = env.reset()
            for t in range(210):
                env.render()
                action_ = 9  # Choose an initial action for demonstration purposes
                # action_ = np.random.randint(0, 16)  # Updated from 4 to 16, to match the new action space

                action = discrete_to_continuous_action(action_)
                action_ = action
                obs = preprocess_observation(obs)


                action_queue.append(action)
                obs_queue.append(obs)


                if len(obs_queue) > K:

                    current_obs = obs_queue[0]
                    current_action = action_queue[0]
                    future_obs = obs_queue[-1]

                    # obs_ = obs
                    # current = obs_
                        
                    goal_obs, humans_obs, tables_obs, plants_obs = transform_processed_observation_into_raw(current_obs)
                    world_image_new = (np.ones((int(RESOLUTION_Y),int(RESOLUTION_X),3))*255).astype(np.uint8)
                    input_grey = draw_observation(world_image_new, humans_obs, tables_obs, plants_obs, goal_obs)

                    ############################################################################################# Pridicted timestep ######################################################################################################
                    CURRENT_OBS = normalize_observation(current_obs)
                    ACTION = normalize_action(current_action)
            
                    states = torch.cat([CURRENT_OBS, torch.from_numpy(ACTION)], dim=-1) # (B, T, vsize+asize
                    states = states.unsqueeze(0).float() .to(device)
                    predicted_obs_, _, _, next_hidden =  rnn.infer(states.unsqueeze(0), next_hidden)               
                    predicted_obs_ = predicted_obs_.squeeze(0)          
                    predicted_obs = predicted_obs_[-1, :]
                    predicted_obs = denormalize_observation(predicted_obs)
                    goal_obs_o, humans_obs_o, tables_obs_o, plants_obs_o = transform_processed_observation_into_raw(predicted_obs.squeeze(0).cpu().detach())
                    predict_world = (np.ones((int(RESOLUTION_Y),int(RESOLUTION_X),3))*255).astype(np.uint8)
                    output_grey = draw_observation(predict_world, humans_obs_o, tables_obs_o, plants_obs_o, goal_obs_o)


                    goal_obs_, humans_obs_, tables_obs_, plants_obs_ = transform_processed_observation_into_raw(future_obs.squeeze(0).cpu())
                    next_image_new = (np.ones((int(RESOLUTION_Y),int(RESOLUTION_X),3))*255).astype(np.uint8)
                    next_timestep_grey = draw_observation(next_image_new, humans_obs_, tables_obs_, plants_obs_, goal_obs_)

                    merged = cv2.merge([next_timestep_grey, next_timestep_grey, output_grey])
                    cv2.imshow(" RobotFrame  Timestep 1 window slide 199", merged)
            



                # action = torch.from_numpy(action_).float()
                nxt_obs, nxt_reward, done, _ = env.step(action_)
                obs = nxt_obs
                t+=1
                if done:
                    print("Episode [{}/{}] finished after {} timesteps".format(s + 1, self.num_episodes_to_record, t), flush=True)
                    obs = env.reset()
                    s+=1

                    break
                

if __name__ == "__main__":

    env = gym.make("SocNavEnv-v1")
    env.configure('./configs/env_timestep_1.yaml')
    env.set_padded_observations(True)
    rollout_dic = {}
    rollout_dir = 'Data/'
    total_episodes = int(input("Please enter the number of episodes: "))
    train_dataset = Rollout(rollout_dic, rollout_dir, 'train', total_episodes)
    train_dataset.make_rollout()
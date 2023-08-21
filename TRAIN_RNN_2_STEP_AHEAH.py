import torch
import torch.nn as nn
import numpy as np
from hparams import RNNHyperParams as hp
from sklearn.metrics import mean_absolute_error, mean_squared_error
from hparams import RobotFrame_Continuous_Datasets_Timestep_1 as data #RobotFrame_Continuous_Datasets_Timestep_1
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hparams import Seq_Len as Seq_len
from torch.utils.data import DataLoader
from data import *
from tqdm import tqdm
import os, sys
from torch.nn import functional as F
from datetime import datetime


from torch.utils.tensorboard import SummaryWriter
import yaml
from UTILITY.early_stopping_for_rnn import  EarlyStopping

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


class RNN_MODEL():
    def __init__(self, config:str, **kwargs) -> None:
        assert(config is not None)
        # initializing the env
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extra = None
        # self.data_dir = None
        self.extra_dir = None
        self.ckpt_dir = None
        # rnn variables
        self.n_latents = None
        self.input_dim = None
        self.n_actions = None
        self.n_hiddens = None
        self.batch_size = None
        self.test_batch = None
        self.n_sample = None
        self.log_interval = None
        self.save_interval = None
        self.max_step = None
        self.save_path = None
        self.n_workers = None
        self.run_name = None
        self.n_layers = None
        self.run_name = None
        self.window = Seq_len.seq_199
        # setting values from config file
        self.configure(self.config)
        # declaring the network
        global_step = 0

        # self.vae = VAE(self.input_dim,256,self.n_latents).to(DEVICE)
        print("self.n_hiddens",self.n_hiddens)
        print("self.n_latents",self.n_latents)

        self.rnn = RNN(self.n_latents, self.n_actions, self.n_hiddens, self.n_layers).to(DEVICE)
        self.ckpt_dir = data.ckpt_dir#'ckpt'

        self.ckpt_dir = data.ckpt_dir#'ckpt'
        self.rnnsave = data.rnnsave#'ckpt'
        self.data_path = data.data_dir 
        self.seq_len = Seq_len.seq_len_199
        episode_length = data.time_steps

        print(self.seq_len) 
        self.ckpt_dir = os.path.join(self.ckpt_dir, self.rnnsave + self.window + "_2_STEPAHEAD")


        dataset = GameEpisodeDatasetNonPrePadded(self.data_path, seq_len=self.seq_len,episode_length=episode_length)

        self.loader = DataLoader(
            dataset, batch_size=16, shuffle=True, drop_last=True,
            num_workers=self.n_workers, collate_fn=collate_fn)
        testset = GameEpisodeDatasetNonPrePadded(self.data_path, seq_len=self.seq_len, training=False,episode_length=episode_length)
        self.valid_loader = DataLoader(
            testset, batch_size=1, shuffle=False, drop_last=False, collate_fn=collate_fn)

        sample_dir = os.path.join(self.ckpt_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=0.0001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.2, verbose=True)

    def configure(self, config:str):
        with open(config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)

        if self.extra is None:
            self.extra = config["extra"]
            assert(self.extra is not None), f"Argument seq_len size cannot be None"

        if self.extra_dir is None:
            self.extra_dir = config["extra_dir"]
            assert(self.extra_dir is not None), f"Argument extra_dir cannot be None"

        if self.ckpt_dir is None:
            self.ckpt_dir = config["ckpt_dir"]
            assert(self.ckpt_dir is not None), f"Argument ckpt_dir  cannot be None"

        if self.n_latents is None:
            self.n_latents = config["n_latents"]
            assert(self.n_latents is not None), f"Argument n_latents size cannot be None"

        if self.input_dim is None:
            self.input_dim = config["input_dims"]
            assert(self.input_dim is not None), f"Argument input_dims size cannot be None"

        if self.n_hiddens is None:
            self.n_hiddens = config["n_hiddens"]
            assert(self.n_hiddens is not None), f"Argument hidden_layers cannot be None"

        if self.n_actions is None:
            self.n_actions = config["n_actions"]
            assert(self.n_actions is not None), f"Argument n_actions cannot be None"

        if self.batch_size is None:
            self.batch_size = config["batch_size"]
            assert(self.batch_size is not None), f"Argument batch_size cannot be None"

        if self.test_batch is None:
            self.test_batch = config["test_batch"]
            assert(self.test_batch is not None), f"Argument test_batch cannot be None"

        if self.n_sample is None:
            self.n_sample = config["n_sample"]
            assert(self.n_sample is not None), f"Argument n_sample cannot be None"

        if self.log_interval is None:
            self.log_interval = config["log_interval"]
            assert(self.log_interval is not None), f"Argument log_interval cannot be None"


        if self.save_interval is None:
            self.save_interval = config["save_interval"]
            assert(self.save_interval is not None), f"Argument save_interval cannot be None"


        if self.max_step is None:
            self.max_step = config["max_step"]
            assert(self.max_step is not None), f"Argument max_step cannot be None"

        if self.save_path is None:
            self.save_path = config["save_path"]
            assert(self.save_path is not None), f"Argument save_path cannot be None"

        if self.n_workers is None:
            self.n_workers = config["n_workers"]
            assert(self.n_workers is not None), f"Argument n_workers cannot be None"

        if self.run_name is None:
            self.run_name = config["run_name"]
            assert(self.run_name is not None), f"Argument run_name cannot be None"

        if self.n_layers is None:
            self.n_layers = config["n_layers"]
            assert(self.n_layers is not None), f"Argument n_layers cannot be None"

        RNN_runs = data.RNN_runs
        if not os.path.exists(RNN_runs ):
            os.makedirs(RNN_runs)
        if self.run_name is not None:
            self.writer = SummaryWriter('RNN_model_runs/'+RNN_runs  + self.window +"_2_STEPAHEAD" )
        else:
            self.writer = SummaryWriter()

        self.early_stopping = EarlyStopping(patience=20, verbose=True)
        self.best_score = 0




    def plot(self, episode):
        self.Train_loss.append(self.train_loss)
        self.Valid_loss.append(self.valid_loss)
        self.grad_norms.append(self.total_grad_norm/self.batch_size)

        if not os.path.isdir(os.path.join(self.save_path, "plots")):
            os.makedirs(os.path.join(self.save_path, "plots"))


        np.save(os.path.join(self.save_path, "plots", "grad_norms"), np.array(self.total_grad_norm/self.batch_size), allow_pickle=True, fix_imports=True)

        np.save(os.path.join(self.save_path, "plots", "Train_loss"), np.array(self.train_loss), allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.save_path, "plots", "Valid_loss"), np.array(self.valid_loss), allow_pickle=True, fix_imports=True)

        self.writer.add_scalar("Train_loss / epoch", self.train_loss, episode)
        self.writer.add_scalar("valid_loss / epoch", self.valid_loss, episode)
        self.writer.add_scalar("Average total grad norm / episode", (self.total_grad_norm/self.batch_size), episode)




    def train(self):
        self.Train_loss = []
        self.Valid_loss = []
        self.grad_norms = []
        # to track the validation loss as the model trains
        self.global_step = 0
        self.train_losses = []
        self.valid_losses = []

        self.l1 = nn.L1Loss()

        def evaluate(self):
            self.rnn.eval()
            self.total_loss = []
            self.val_mae = []
            self.val_rmse = []
            self.total_loss = []
            with torch.no_grad():
                for idx, (obs, actions) in enumerate(self.valid_loader):
                    obs = normalize_observation(obs)
                    actions = normalize_action(actions)
        
                    obs, actions = obs.to(DEVICE), actions.to(DEVICE)
                    # znew,latent_mu, latent_var ,z = self.vae(obs) # (B*T, vsize)
                    z = obs
    
                    z = z.view(-1, self.seq_len, self.n_latents) # (B*n_seq, T, vsize)
                    actions = actions.view(-1, self.seq_len, self.n_actions) # (B*n_seq, T, vsize)
                    next_z = z[:, 2:, :]
                    z, actions = z[:, :-2, :], actions[:, :-2, :]
                    states = torch.cat([z, actions], dim=-1) # (B, T, vsize+asize)
                    x, _, _ = self.rnn(states)
                    loss = self.l1(x, next_z)
                    mae = mean_absolute_error(next_z.cpu().detach().numpy().reshape(-1, next_z.shape[2]), x.cpu().detach().numpy().reshape(-1, x.shape[2]))
                    rmse = np.sqrt(mean_squared_error(next_z.cpu().detach().numpy().reshape(-1, next_z.shape[2]),x.cpu().detach().numpy().reshape(-1, x.shape[2])))
                    self.val_mae.append(mae)
                    self.val_rmse.append(rmse)

            self.total_loss.append(loss.item())
            self.rnn.train()
            return np.mean(self.total_loss), np.mean(self.val_mae), np.mean(self.val_rmse)

        for idx in range(1, self.max_step + 1):        # while self.global_step < self.max_step:
            self.Train_loss = []
            self.Valid_loss = []
            self.grad_norms = []
            self.train_losses = []
            self.valid_losses = []
            self.total_grad_norm = 0  

            for idx, (obs, actions) in enumerate(self.loader):
                with torch.no_grad():
                    obs, actions = obs.to(DEVICE), actions.to(DEVICE)
                    obs = normalize_observation(obs)
                    actions = normalize_action(actions)
                    z = obs
                    z = z.view(-1, self.seq_len, self.n_latents) # (B*n_seq, T, vsize)
                    actions = actions.view(-1, self.seq_len, self.n_actions) # (B*n_seq, T, vsize)

                next_z = z[:, 2:, :]
                z, actions = z[:, :-2, :], actions[:, :-2, :]      
                states = torch.cat([z, actions], dim=-1) # (B, T, vsize+asize)
                x, _, _ = self.rnn(states)
                loss = self.l1(x, next_z)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.train_losses.append(loss.item())
            self.global_step += 1

            self.total_grad_norm += (torch.nn.utils.clip_grad_norm_(self.rnn.parameters(), max_norm=0.5).cpu())/self.global_step
            avg_train_loss = np.mean(self.train_losses)
            avg_train_mae = mean_absolute_error(next_z.cpu().detach().numpy().reshape(-1, next_z.shape[2]), x.cpu().detach().numpy().reshape(-1, x.shape[2]))
            avg_train_rmse = np.sqrt(mean_squared_error(next_z.cpu().detach().numpy().reshape(-1, next_z.shape[2]),x.cpu().detach().numpy().reshape(-1, x.shape[2])))

            if self.global_step % 1 == 0:
                self.valid_losses, avg_val_mae, avg_val_rmse = evaluate(self)
                print(f"Epoch {self.global_step}, Train Loss: {avg_train_loss:.4f}, Val Loss: {self.valid_losses:.4f}, Train MAE: {avg_train_mae:.4f}, Val MAE: {avg_val_mae:.4f}, Train RMSE: {avg_train_rmse:.4f}, Val RMSE: {avg_val_rmse:.4f}")
                
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(os.path.join(self.ckpt_dir, 'train.log'), 'a') as f:
                    log = '{} || Step: {}, train_loss: {:.4f}, loss: {:.4f}\n'.format(now, self.global_step, loss.item(), self.valid_losses)
                    f.write(log)

            self.epoch_len = len(str(self.global_step))
            self.train_loss = np.mean(self.train_losses)#/len(self.loader)
            self.valid_loss = self.valid_losses#/len(self.valid_loader)
            self.scheduler.step(self.valid_loss)  # Updating the scheduler

            self.plot(self.global_step +1)

            print_msg = (f'[{self.global_step:>{self.epoch_len}}/{self.global_step:>{self.epoch_len}}] ' +
                        f'train_loss: {self.train_loss:.8f} ' +
                        f'valid_loss: {self.valid_loss:.8f}')
            

                # clear lists to track next epoch
            self.train_losses = []
            self.valid_losses = []

            if self.global_step % 1 == 0:
                d = {
                    'model': self.rnn.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }
                torch.save(
                    d, os.path.join(self.ckpt_dir, '{:03d}two.pth.tar'.format(self.global_step//self.save_interval )))


                # and if it has, it will make a checkpoint of the current model
            if self.global_step % 1 == 0:
                self.early_stopping(self.valid_loss, self.rnn)



            if self.global_step % 5 == 0:
                print(print_msg)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # np.random.seed(0)

    # config file for the model
    config = "./configs/Robotframe_RNN_modelfull.yaml"
        # declaring the network
    Agent =RNN_MODEL(config, run_name="RNN_model_runs")

    Agent.train()
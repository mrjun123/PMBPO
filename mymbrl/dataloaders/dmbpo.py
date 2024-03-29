import torch
import torch.utils.data as data
import numpy as np

class DMBPO(data.Dataset):
    def __init__(self, device):
        self.data = []
        self.data_len = 0
        self.break_dict = {}
        self.device = device

    def empty(self):
        self.data = []
        self.data_len = 0
        
    def __len__(self) -> int:
        return self.data_len
    
    def len(self) -> int:
        return self.data_len
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        return f'Dynamics Data Buffer with {len(self.data)} / {self.capacity} elements.\n'

    def shuffle(self):
        pass
        # idxs = np.random.randint(self.x.shape[0], size=[self.model.num_nets, self.train_in.shape[0]])

    def get_x_y_all(self):
        obs_all = []
        acs_all = []
        next_obs_all = []
        for i in range(self.data_len):
            obs, acs, next_obs = self.data[i]
            obs_all.append(obs)
            acs_all.append(acs)
            next_obs_all.append(next_obs)
        return torch.tensor(obs_all, dtype=torch.float32, device=self.device), torch.tensor(acs_all, dtype=torch.float32, device=self.device), torch.tensor(next_obs_all, dtype=torch.float32, device=self.device)
    
    def push(self, obs, acs, next_obs, path_done=True):
        new_data_len = obs.shape[0]
        data_index = self.data_len

        for i in range(obs.shape[0]):
            self.data.append((obs[i], acs[i], next_obs[i]))
        if path_done:
            self.break_dict[data_index + new_data_len - 1] = True
        self.data_len += new_data_len

    def set_hold_idxs(self, hold_idxs):
        self.hold_dict = {}
        for i in range(hold_idxs.shape[0]):
            self.hold_dict[hold_idxs[i]] = True

    def next_index(self, index):
        if index >= self.data_len - 1:
            return -1
        if index in self.break_dict:
            return -1
        if (index + 1) in self.hold_dict:
            return -1
        if index == -1:
            return index
        return index + 1
    
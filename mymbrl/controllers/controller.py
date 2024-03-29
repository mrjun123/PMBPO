import torch
import numpy as np
from mymbrl.utils import has_input_param
from mymbrl.envs.utils import termination_fn

class Controller:
    def __init__(self, agent, is_torch=True, writer=None):
        self.config = agent.config
        self.is_torch = is_torch
        self.agent = agent
        self.prediction = agent.prediction
        self.env = agent.env
        self.dataloader = agent.dataloader
        self.writer = writer
        self.exp_epoch = -1

        self.mpc_obs = [0] * (self.config.agent.predict_length + 1)
        self.mpc_acs = [0] * self.config.agent.predict_length
        self.mpc_nonterm_masks = [0] * self.config.agent.predict_length

        self.net_weight = torch.ones(self.config.agent.elite_size, device=self.config.device)
        
    def set_epoch(self, exp_epoch):
        self.exp_epoch = exp_epoch
        
    def set_step(self, exp_step):
        self.exp_step = exp_step
        
    def add_data_step(self, cur_state, action, reward, next_state, done):
        pass
    def add_two_step_data(self, pre_state, pre_action, state, action, reward, next_state, done, is_start):
        pass

    def train_epoch(self, epoch_reward):
        pass
        
    def train_step(self):
        pass
        
    def sample(self, states, epoch=-1, step=-1):
        self.epoch = epoch
        self.step = step

    
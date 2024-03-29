import numpy as np
import gym
import torch
from torch import nn as nn
from torch.nn import functional as F
from mymbrl.utils import swish, get_affine_params
import random
import math

def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)
class DropoutBbpoModel(nn.Module):

    def __init__(self, ensemble_size, in_features, out_features, hidden_size=200,drop_prob=0.1, dropout_mask_nums=20, device="cpu"):
        super().__init__()

        self.fit_input = False
        self.dropout = False
        self.mask_batch_size = 1
        self.batch_size = 30

        self.num_nets = ensemble_size

        self.drop_prob = drop_prob
        self.dropout_mask_nums = dropout_mask_nums
        self.hidden_size = hidden_size
        self.elite_index = None

        self.hidden1_mask = None
        self.hidden2_mask = None
        self.hidden3_mask = None
        self.hidden4_mask = None

        self.hidden_mask_indexs = None
        self.hidden1_mask_select = None
        self.hidden2_mask_select = None
        self.hidden3_mask_select = None
        self.hidden4_mask_select = None

        self.in_features = in_features
        self.out_features = out_features

        self.lin0_w_e = None

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, in_features, hidden_size)
        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, hidden_size, hidden_size)
        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, hidden_size, hidden_size)
        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, hidden_size, hidden_size)
        self.lin4_w, self.lin4_b = get_affine_params(ensemble_size, hidden_size, out_features)

        self.inputs_mu = nn.Parameter(torch.zeros(in_features).to(device), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(in_features).to(device), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32).to(device) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32).to(device) * 10.0)
        self.scaler = StandardScaler()

    def set_elite_index(self, elite_index):
        self.elite_index = elite_index
        self.lin0_w_e, self.lin0_b_e = self.lin0_w[elite_index,:,:], self.lin0_b[elite_index,:,:]
        self.lin1_w_e, self.lin1_b_e = self.lin1_w[elite_index,:,:], self.lin1_b[elite_index,:,:]
        self.lin2_w_e, self.lin2_b_e = self.lin2_w[elite_index,:,:], self.lin2_b[elite_index,:,:]
        self.lin3_w_e, self.lin3_b_e = self.lin3_w[elite_index,:,:], self.lin3_b[elite_index,:,:]
        self.lin4_w_e, self.lin4_b_e = self.lin4_w[elite_index,:,:], self.lin4_b[elite_index,:,:]
        if not self.dropout:
            return
        self.hidden1_mask_select_e = self.hidden1_mask_select[elite_index,:,:]
        self.hidden2_mask_select_e = self.hidden2_mask_select[elite_index,:,:]
        self.hidden3_mask_select_e = self.hidden3_mask_select[elite_index,:,:]
        self.hidden4_mask_select_e = self.hidden4_mask_select[elite_index,:,:]
        
    def compute_decays(self):

        lin0_decays = 0.000025 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.00005 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.000075 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.000075 * (self.lin3_w ** 2).sum() / 2.0
        lin4_decays = 0.0001 * (self.lin4_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays + lin4_decays

    def fit_input_stats(self, data):
        
        device = self.get_param_device()

        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = torch.from_numpy(mu).to(device).float()
        self.inputs_sigma.data = torch.from_numpy(sigma).to(device).float()
        self.fit_input = True

    def forward(self, inputs, ret_logvar=False, return_hidden=False,open_dropout=True):
        inputs = self.scaler.transform(inputs)
        # self.select_mask(inputs.shape[1])
        # Transform inputs
        if self.fit_input:
            inputs = (inputs - self.inputs_mu) / self.inputs_sigma
        
        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden1_mask_select

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden2_mask_select

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden3_mask_select

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b
        
        if return_hidden:
            inputs = AvgL1Norm(inputs)
            return inputs
        
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden4_mask_select
        inputs = inputs.matmul(self.lin4_w) + self.lin4_b 

        mean = inputs[:, :, :self.out_features // 2]
        logvar = inputs[:, :, self.out_features // 2:]

        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)
    
    def elite_forward(self, inputs, ret_logvar=False, open_dropout=True, return_hidden=False):
        inputs = self.scaler.transform(inputs)
        inputs = inputs.matmul(self.lin0_w_e) + self.lin0_b_e
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden1_mask_select_e

        inputs = inputs.matmul(self.lin1_w_e) + self.lin1_b_e
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden2_mask_select_e

        inputs = inputs.matmul(self.lin2_w_e) + self.lin2_b_e
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden3_mask_select_e

        inputs = inputs.matmul(self.lin3_w_e) + self.lin3_b_e
        
        if return_hidden:
            inputs = AvgL1Norm(inputs)
            return inputs
        inputs = swish(inputs)
        if self.dropout and open_dropout:
            inputs = inputs * self.hidden4_mask_select_e

        inputs = inputs.matmul(self.lin4_w_e) + self.lin4_b_e

        mean = inputs[:, :, :self.out_features // 2]

        logvar = inputs[:, :, self.out_features // 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)
    
    def sample_new_mask(self, dropout_mask_nums=None, num_particles=None):
        """Sample a new mask for MC-Dropout. Rather than sample the mask at each forward pass
        (as traditionally done in dropout), keep the dropped nodes fixed until this function is
        explicitely called.
        """
        self.dropout = True
        drop_prob = self.drop_prob
        device = self.get_param_device()
        if dropout_mask_nums:
            self.dropout_mask_nums = dropout_mask_nums
        # Sample dropout random masks
        self.hidden1_mask = torch.bernoulli(
            torch.ones(self.num_nets, dropout_mask_nums, self.hidden_size) * (1 - drop_prob)).to(device)
        self.hidden2_mask = torch.bernoulli(
            torch.ones(self.num_nets, dropout_mask_nums, self.hidden_size) * (1 - drop_prob)).to(device)
        self.hidden3_mask = torch.bernoulli(
            torch.ones(self.num_nets, dropout_mask_nums, self.hidden_size) * (1 - drop_prob)).to(device)
        self.hidden4_mask = torch.bernoulli(
            torch.ones(self.num_nets, dropout_mask_nums, self.hidden_size) * (1 - drop_prob)).to(device)
    
    def select_mask(self, batch_size=None):
        if not self.dropout:
            return
        self.batch_size = batch_size
        index_list = list(range(0, self.dropout_mask_nums))
        index_tile_list = index_list*math.ceil(batch_size / self.dropout_mask_nums)

        device = self.get_param_device()
        indexs = torch.tensor(random.sample(index_tile_list, batch_size), device=device)

        self.hidden1_mask_select = self.hidden1_mask[:, indexs, :]
        self.hidden2_mask_select = self.hidden2_mask[:, indexs, :]
        self.hidden3_mask_select = self.hidden3_mask[:, indexs, :]
        self.hidden4_mask_select = self.hidden4_mask[:, indexs, :]
        if self.elite_index is not None:
            self.hidden1_mask_select_e = self.hidden1_mask_select[self.elite_index,:,:]
            self.hidden2_mask_select_e = self.hidden2_mask_select[self.elite_index,:,:]
            self.hidden3_mask_select_e = self.hidden3_mask_select[self.elite_index,:,:]
            self.hidden4_mask_select_e = self.hidden4_mask_select[self.elite_index,:,:]
        
    def reset_weights(self):
        device = self.get_param_device()
        self.lin0_w, self.lin0_b = get_affine_params(self.num_nets, self.in_features, self.hidden_size, device=device)
        self.lin1_w, self.lin1_b = get_affine_params(self.num_nets, self.hidden_size, self.hidden_size, device=device)
        self.lin2_w, self.lin2_b = get_affine_params(self.num_nets, self.hidden_size, self.hidden_size, device=device)
        self.lin3_w, self.lin3_b = get_affine_params(self.num_nets, self.hidden_size, self.hidden_size, device=device)
        self.lin4_w, self.lin4_b = get_affine_params(self.num_nets, self.hidden_size, self.out_features, device=device)
        
        self.max_logvar = nn.Parameter(torch.ones(1, self.out_features // 2, dtype=torch.float32).to(device) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, self.out_features // 2, dtype=torch.float32).to(device) * 10.0)
    
    def get_param_device(self):
        return next(self.parameters()).device


class StandardScaler(object):
    def __init__(self):
        self.is_fit = False
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.is_fit = True
        data = data.reshape(-1, data.shape[-1])
        self.mu = torch.mean(data, dim=0, keepdim=True)
        self.std = torch.std(data, dim=0, keepdim=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        if not self.is_fit:
            return data
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu


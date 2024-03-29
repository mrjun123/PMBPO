from .agent import Agent
import mymbrl.controllers as controllers
import mymbrl.models as models
import mymbrl.envs as envs
import mymbrl.dataloaders as dataloaders
import torch, numpy as np
import torch.nn as nn
from scipy.stats import norm
import time
from joblib import Parallel, delayed
import dill
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import parallel_backend
import random
import datetime
import mymbrl.optimizers as optimizers
from mymbrl.utils import shuffle_rows

class RLHomework(Agent):

    def __init__(self, config, env, writer):
        """
        Controller: MPC
        """
        self.config = config
        self.env = env
        self.writer = writer
        self.exp_epoch = 0
        
        Model = models.get_item(config.agent.model)
        
        self.model = Model(
            ensemble_size=config.agent.ensemble_size,
            in_features=env.MODEL_IN,
            out_features=env.MODEL_OUT*2,
            hidden_size=config.agent.dynamics_hidden_size, 
            drop_prob=config.agent.dropout,
            device=config.device
        )
        
        self.model = self.model.to(config.device)
        self.dynamics_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.agent.dynamics_lr, 
            weight_decay=self.config.agent.dynamics_weight_decay
        )
        self.dynamics_scheduler = torch.optim.lr_scheduler.StepLR(
        self.dynamics_optimizer, step_size=1, gamma=config.agent.dynamics_lr_gamma)
        if config.agent.dropout > 0:
            self.model.sample_new_mask(batch_size=config.agent.dropout_mask_nums)

        Controller = controllers.get_item(config.agent.controller)
        self.controller = Controller(
            self.prediction,
            config=config, 
            env = self.env,
            writer=writer
        )

        Dataloader = dataloaders.get_item('free_trend')
        self.dataloader = Dataloader()

    def reset(self):
        self.controller.reset()
        
    def set_epoch(self, exp_epoch):
        
        self.exp_epoch = exp_epoch
        # self.controller.set_epoch(exp_epoch)
    
    def set_step(self, exp_step):
        self.exp_step = exp_step
        # self.controller.set_step(exp_step)
        
    def set_model(self, model):
        self.model = model

    def train(self):
        """
        训练一个agent
        """
        # self.pre_train()
        dynamics_model = self.model
        # if self.config.agent.dropout > 0:
        #     dynamics_model.sample_new_mask(batch_size=self.config.agent.dropout_mask_nums)
        
        for param in dynamics_model.parameters():
            param.requires_grad = True

        
        # dynamics_model.fit_input_stats(self.dataloader.get_x_all())
        
        num_nets = dynamics_model.num_nets
        num_particles = self.config.agent.num_particles
        net_particles = num_particles // num_nets
        
        if self.config.agent.reset_model:
            dynamics_model.reset_weights()
        dynamics_model.train()
        dynamics_optimizer = self.dynamics_optimizer
        
        self.trainloader = torch.utils.data.DataLoader(
            self.dataloader, 
            batch_size=self.config.agent.train_batch_size, 
            shuffle=True,
            drop_last=True
        )
        
        i = 0

        data_len = self.dataloader.len()
        if self.config.agent.dynamics_type == "bootstrap":
            idxs = np.random.randint(data_len, size=[num_nets, data_len])
        elif self.config.agent.dynamics_type == "naive":
            idxs = np.arange(data_len)
            idxs = idxs.reshape(1,-1)
            idxs = np.tile(idxs, [num_nets, 1])
        
        batch_size = self.config.agent.train_batch_size
        num_batch = idxs.shape[-1] // batch_size

        x_all, y_all, a_all, y2_all, x2_all = self.dataloader.get_x_y_all()
        for _ in range(self.config.agent.train_epoch):
            idxs = shuffle_rows(idxs)
            for batch_num in range(num_batch):
                if self.config.agent.mc:
                    dynamics_model.sample_new_mask(batch_size=30)
                dynamics_model.select_mask(30)
                batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]
                # self.dataloader[batch_idxs]
                # print('x_all shape', x_all.shape)
                # print('batch_idxs shape', batch_idxs.shape)
                
                x, y, a, y2, x2 = x_all[batch_idxs, :], y_all[batch_idxs, :], a_all[batch_idxs, :], y2_all[batch_idxs, :], x2_all[batch_idxs, :]
                x, y, a, y2, x2 = x.to(self.config.device), y.to(self.config.device), a.to(self.config.device), y2.to(self.config.device), x2.to(self.config.device)
                
                loss = self.config.agent.dynamics_weight_decay_rate * dynamics_model.compute_decays()
                loss += 0.005 * (dynamics_model.max_logvar.sum() - dynamics_model.min_logvar.sum())
                mean, logvar = dynamics_model(x, ret_logvar=True)
                if self.config.agent.tau > 0 :
                    logvar = torch.log(torch.ones_like(mean, device=self.config.device)*self.config.agent.tau)

                inv_var = torch.exp(-logvar)
                mes_loss = ((mean - y) ** 2)
                mes_loss_sum = mes_loss.mean(-1).mean(-1).sum()
                train_losses = mes_loss * inv_var + logvar
                # if self.exp_epoch > 10:
                #     train_losses = torch.where(train_losses < 100, train_losses, mes_loss)
                train_losses = train_losses.mean(-1).mean(-1).sum()
                loss += train_losses

                # y_2_true = self.env.obs_postproc(x2, y2)
                # new_y2 = self.env.targ_proc(inputs2, y_2_true)
                if self.config.agent.fitting_error_correction:
                    # 获取真实值
                    predictions = self.env.obs_postproc(x2, mean)
                    predictions_true = self.env.obs_postproc(x2, y)
                    # 对输入数据进行处理
                    predictions_preproc = self.env.obs_preproc(predictions)
                    # action = self._expand_to_ts_format(action)
                    inputs2 = torch.cat((predictions_preproc, a), dim=-1)
                    if self.config.agent.mc:
                        dynamics_model.sample_new_mask(batch_size=30)
                    dynamics_model.select_mask(30)
                    mean2, logvar2 = dynamics_model(inputs2, ret_logvar=True)
                    if self.config.agent.tau > 0 :
                        logvar2 = torch.log(torch.ones_like(mean2, device=self.config.device)*self.config.agent.tau)

                    y_2_true = self.env.obs_postproc(predictions_true, y2)
                    new_y2 = self.env.targ_proc(predictions, y_2_true)

                    inv_var2 = torch.exp(-logvar2)
                    mes_loss2 = ((mean2 - new_y2) ** 2)
                    mes_loss_sum2 = mes_loss2.mean(-1).mean(-1).sum()
                    train_losses2 = mes_loss2 * inv_var2 + logvar2
                    # if self.exp_epoch > 10:
                    #     train_losses2 = torch.where(train_losses2 < 100, train_losses2, mes_loss2)
                    train_losses2 = train_losses2.mean(-1).mean(-1).sum()
                    loss += train_losses2

                loss.backward()
                nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=20, norm_type=2)
                dynamics_optimizer.step()
                dynamics_optimizer.zero_grad()
                i += 1
                if i % 50 == 0:
                    if self.config.agent.fitting_error_correction:
                        print(i, 'train_loss', loss.item(), 'MESLoss', mes_loss_sum.item(), 'MESLoss2', mes_loss_sum2.item())
                    else:
                        print(i, 'train_loss', loss.item(), 'MESLoss', mes_loss_sum.item())
        
        # if self.config.agent.optimizer == 'CEM':
        #     self.model.select_mask(net_particles, mask_batch_size=self.config.agent.CEM.popsize)
        # elif self.config.agent.optimizer == 'Adam':  
        #     self.model.select_mask(net_particles)
        
        if self.exp_epoch in self.config.agent.lr_scheduler:
            self.dynamics_scheduler.step()
        self.model.select_mask(net_particles)
        for param in dynamics_model.parameters():
            param.requires_grad = False
        print('max_var', dynamics_model.max_logvar.exp().mean())
        print('min_var', dynamics_model.min_logvar.exp().mean())
    
    def sample(self, states):
        """
        根据当前环境获取一个动作
        """
        # num_nets = self.config.agent.ensemble_size
        # num_particles = self.config.agent.num_particles
        # net_particles = num_particles // num_nets
        # self.model.select_mask(net_particles)
        self.model.eval()
        # for param in self.model.parameters():
        #     param.requires_grad = False
        action = self.controller.sample(states, self.exp_epoch, self.exp_step)
        # for param in self.model.parameters():
        #     param.requires_grad = True
        print('action', action.max(), action.min())
        return action

    def add_data(self, states, actions, indexs=[]):
        assert states.shape[0] == actions.shape[0] + 1
        x = np.concatenate((self.env.obs_preproc(states[:-2]), actions[:-1]), axis=1)
        y = self.env.targ_proc(states[:-2], states[1:-1])
        a = actions[1:]
        y2 = self.env.targ_proc(states[1:-1], states[2:])
        x2 = states[:-2]

        self.dataloader.push(x, y, a, y2, x2)
        # if not os.path.exists(os.path.join(self.config.run_dir,'data')):
        #     os.mkdir(os.path.join(self.config.run_dir,'data'))
        # with open(os.path.join(self.config.run_dir,'data', str(self.exp_epoch)+'.pkl'),'wb') as f:
        #     dill.dump(self.dataloader, f)
    
    # @torch.compile
    def prediction(self, states, action, t=0, sample_epoch=0, print_info=False, return_reward_states=False):
        
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.config.device).float()
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, device=self.config.device).float()
        if(states.dim() == 1):
            states = states.unsqueeze(0).tile(self.config.agent.num_particles, 1).float()
            # states = states.unsqueeze(0).expand(self.config.agent.num_particles, states.shape[-1]).float()
        if(action.dim() == 1):
            # action = action.unsqueeze(0).expand(states.shape[0], action.shape[-1]).float()
            action = action.unsqueeze(0).tile(states.shape[0], 1).float()

        proc_obs = self.env.obs_preproc(states)

        proc_obs = self._expand_to_ts_format(proc_obs)
        action = self._expand_to_ts_format(action)

        inputs = torch.cat((proc_obs, action), dim=-1)
        # model = self.model
        # model = torch.compile(model)

        num_nets = self.model.num_nets
        num_particles = self.config.agent.num_particles
        net_particles = num_particles // num_nets
        net_particles_batch = inputs.shape[1]
        batch_size = net_particles_batch // net_particles
        if self.config.agent.dropout_remask:
            if self.config.agent.mc:
                self.model.sample_new_mask(batch_size=net_particles)
            self.model.select_mask(net_particles)
        
        self.model.set_mask_batch(batch_size)
        mean, var = self.model(inputs)
        old_mu = torch.mean(mean, dim=1, keepdim=True)
        sigma_old = torch.std(mean, dim=1, keepdim=True)

        if self.config.agent.aleatoric == 'dpets1':
            if t == 0:
                predictions = mean + torch.randn_like(mean, device=self.config.device) * var.sqrt()
            else:
                predictions = mean
        elif self.config.agent.aleatoric == 'dpilco':
            mu1 = torch.mean(mean, dim=0, keepdim=True)
            mu1 = mu1.expand(mean.shape[0], net_particles, mean.shape[-1])
            sigma1 = torch.std(mean, dim=0, keepdim=True)
            sigma1 = sigma1.expand(mean.shape[0],net_particles,mean.shape[-1])
            predictions = mu1 + torch.randn_like(mu1, device=self.config.device) * (sigma1+self.config.agent.tau)
            # predictions = mean + torch.randn_like(mean, device=self.config.device) * self.config.agent.tau
        
        elif self.config.agent.aleatoric == 'no':
            predictions = mean
        elif self.config.agent.aleatoric == 'pets':
            predictions = mean + torch.randn_like(mean, device=self.config.device) * var.sqrt()
        elif self.config.agent.aleatoric == 'mm1':
            mu1 = torch.mean(mean, dim=0, keepdim=True)
            mu1 = mu1.expand(mean.shape[0], net_particles, mean.shape[-1])
            sigma1 = torch.std(mean, dim=0, keepdim=True)
            sigma1 = sigma1.expand(mean.shape[0],net_particles,mean.shape[-1])
            predictions = mu1 + torch.randn_like(mu1, device=self.config.device) * sigma1

        elif self.config.agent.aleatoric == 'mm2':
            # mean: num_nets, particles, dim
            mu2 = mean.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
            mu2 = mu2.expand(mean.shape[0], mean.shape[1], mean.shape[-1])
            predictions = mu2
            # sigma1 = torch.std(mean, dim=0, keepdim=True)
            # sigma1 = sigma1.expand(mean.shape[0],net_particles,mean.shape[-1])
            # predictions = mu1 + torch.randn_like(mu1, device=self.config.device) * sigma1
        elif self.config.agent.aleatoric == 'mm3':
            # num_nets, batch_size, net_particles, dim
            temp_mean = mean.reshape(num_nets, batch_size, net_particles, mean.shape[-1])
            temp_mean = temp_mean.mean(dim=2, keepdim=True).tile(1,1,net_particles,1)
            temp_mean = temp_mean.reshape(num_nets, -1, mean.shape[-1])
            predictions = temp_mean
        else:
            predictions = mean
        
        
        if print_info and (t+1)%5 == 0:
            print('pred_step', t, 'dropout_std', sigma_old.mean().item(), 'pets_std', var.sqrt().mean().item(),'bootstrap_std', old_mu.std(dim=0).mean().item())
        
        predictions = self._flatten_to_matrix(predictions)
        if return_reward_states:
            predictions_reward = mean + torch.randn_like(mean, device=self.config.device) * var.sqrt()
            predictions_reward = self._flatten_to_matrix(predictions_reward)
            return self.env.obs_postproc(states, predictions), self.env.obs_postproc(states, predictions_reward)
        return self.env.obs_postproc(states, predictions)
    
    def _expand_to_ts_format(self, mat):
        # input:  num_particles, dim
        # output: num_nets, batch_size * net_particles, dim
        dim = mat.shape[-1]
        # 2: batch_size
        # Before, [10, 5] in case of proc_obs
        reshaped = mat.view(-1, self.model.num_nets, self.config.agent.num_particles // self.model.num_nets, dim)
        # After, [2, 5, 1, 5]
        transposed = reshaped.transpose(0, 1)
        # After, [5, 2, 1, 5]
        reshaped = transposed.contiguous().view(self.model.num_nets, -1, dim)
        # After. [5, 2, 5]
        return reshaped
    

    def _flatten_to_matrix(self, ts_fmt_arr):

        dim = ts_fmt_arr.shape[-1]
        reshaped = ts_fmt_arr.view(self.model.num_nets, -1, self.config.agent.num_particles // self.model.num_nets, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(-1, dim)
        # batch_size * num_nets * net_particles, action_dim
        return reshaped
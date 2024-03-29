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
from mymbrl.utils import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import parallel_backend
import random
import datetime
import mymbrl.optimizers as optimizers
from mymbrl.utils import shuffle_rows
import itertools
import inspect

class DropoutMbpo(Agent):

    def __init__(self, config, env, writer):
        """
        Controller: MPC
        """
        self.config = config
        self.env = env
        self.writer = writer
        self.exp_epoch = 0
        self._max_epochs_since_update = config.agent.max_epochs_since_update
        
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

        if config.agent.dynamics_error_model:
            self.error_model = Model(
                ensemble_size=config.agent.ensemble_size,
                in_features=env.MODEL_IN - env.action_space.shape[0],
                out_features=env.MODEL_IN - env.action_space.shape[0],
                hidden_size=config.agent.dynamics_hidden_size, 
                drop_prob=0,
                device=config.device
            )
            self.error_model = self.error_model.to(config.device)

        self.dynamics_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.agent.dynamics_lr, 
            weight_decay=self.config.agent.dynamics_weight_decay
        )
        self.dynamics_scheduler = torch.optim.lr_scheduler.StepLR(
            self.dynamics_optimizer, 
            step_size=1, 
            gamma=config.agent.dynamics_lr_gamma
        )
        if config.agent.dropout > 0:
            self.model.sample_new_mask(dropout_mask_nums=config.agent.dropout_mask_nums)
            num_nets = self.model.num_nets
            num_particles = self.config.agent.num_particles
            net_particles = num_particles // num_nets
            self.model.select_mask(net_particles)

        Dataloader = dataloaders.get_item('dmbpo')
        self.dataloader = Dataloader(device=self.config.device)

        # HoldDataloader = dataloaders.get_item('free_capacity')
        # self.hold_dataloader = HoldDataloader()

        self._epochs_since_update = 0
        self.improvement_rate = config.agent.improvement_rate
        
        Controller = controllers.get_item(config.agent.controller)
        self.controller = Controller(
            self,
            writer=writer
        )

        self.elite_model_idxes = np.arange(self.config.agent.elite_size).tolist()
        self.model.set_elite_index(self.elite_model_idxes)
        
        

    def batch_loss(self, obs, acs, next_obs, acs2, next_obs2):
        # return
        dynamics_model = self.model
        dynamics_model.train()
        for param in dynamics_model.parameters():
            param.requires_grad = True
        num_nets = dynamics_model.num_nets
        obs = obs.unsqueeze(0).expand(num_nets, -1, -1)
        acs = acs.unsqueeze(0).expand(num_nets, -1, -1)
        next_obs = next_obs.unsqueeze(0).expand(num_nets, -1, -1)
        acs2 = acs2.unsqueeze(0).expand(num_nets, -1, -1)
        next_obs2 = next_obs2.unsqueeze(0).expand(num_nets, -1, -1)
        if self.model.lin0_w_e is None:
            self.model.set_elite_index(self.elite_model_idxes)

        loss = self.config.agent.dynamics_weight_decay_rate * dynamics_model.compute_decays()
        loss += 0.01 * (dynamics_model.max_logvar.sum() - dynamics_model.min_logvar.sum())
        
        x1 = torch.cat([self.env.obs_preproc(obs), acs], dim=-1)
        y1 = self.env.targ_proc(obs, next_obs)
        
        mean, logvar = dynamics_model(x1, ret_logvar=True)
        inv_var = torch.exp(-logvar)
        mes_loss = ((mean - y1) ** 2)
        train_losses = mes_loss * inv_var + logvar
        
        if self.config.agent.fitting_error_correction:
            fec_length = self.config.agent.fec_length

            for l in range(fec_length-1):
                
                signature = inspect.signature(self.env.obs_postproc)
                if 'acs' in signature.parameters:
                    predictions = self.env.obs_postproc(obs, mean, acs2)
                else:
                    predictions = self.env.obs_postproc(obs, mean)
                # predictions = self.env.obs_postproc(obs, mean)
                # predictions = predictions[next_idxs>=0, :]
                predictions_preproc = self.env.obs_preproc(predictions)

                inputs2 = torch.cat((predictions_preproc, acs2), dim=-1)

                mean2, logvar2 = dynamics_model(inputs2, ret_logvar=True)
                if self.config.agent.fec_delta:
                    y2 = self.env.targ_proc(next_obs, next_obs2)
                else:
                    y2 = self.env.targ_proc(predictions, next_obs2)
                inv_var2 = torch.exp(-logvar2)
                mes_loss2 = ((mean2 - y2) ** 2)
                train_losses2 = mes_loss2 * inv_var2 + logvar2

                if self.config.agent.penalty:
                    train_losses2 += self.config.agent.penalty_rate/inv_var2

                train_losses2 = train_losses2 * (self.config.agent.fec_decay ** (l+1))
                # train_losses2 = train_losses2.mean(-1).mean(-1).sum()
                if self.config.agent.fec_logsumexp:
                    train_losses = torch.logsumexp(torch.cat([train_losses.unsqueeze(0), train_losses2.unsqueeze(0)], dim=0), dim=0)
                else:
                    train_losses += train_losses2
        
        train_losses = train_losses.mean(-1).mean(-1).sum()
        loss += train_losses
        loss += train_losses
        loss.backward()
        self.dynamics_optimizer.step()
        self.dynamics_optimizer.zero_grad()
        for param in dynamics_model.parameters():
            param.requires_grad = False
    
    def train(self, return_log=False, epoch_reward=0):
        """
        训练一个agent
        """

        holdout_ratio = self.config.agent.holdout_ratio
        dynamics_model = self.model
        num_nets = dynamics_model.num_nets
        num_particles = self.config.agent.num_particles
        net_particles = num_particles // num_nets

        self._snapshots = {i: (None, 1e10) for i in range(num_nets)}
        
        for param in dynamics_model.parameters():
            param.requires_grad = True
        
        if self.config.agent.reset_model:
            raise ValueError("reset_model is fail in mbpo")
        dynamics_model.train()

        i = 0

        all_data_len = self.dataloader.len()
        num_holdout = int(all_data_len * holdout_ratio)
        data_len = all_data_len - num_holdout

        indices = np.random.permutation(all_data_len)
        wo_hold_indices = indices[:data_len]
        hold_idxs = indices[data_len:]
        self.dataloader.set_hold_idxs(hold_idxs)
        
        if self.config.agent.dynamics_type == "bootstrap":
            all_idxs = np.random.randint(data_len, size=[num_nets, data_len])
        elif self.config.agent.dynamics_type == "naive":
            all_idxs = np.arange(data_len)
            all_idxs = all_idxs.reshape(1,-1)
            all_idxs = np.tile(all_idxs, [num_nets, 1])
        
        idxs = wo_hold_indices[all_idxs]

        batch_size = self.config.agent.train_batch_size
        num_batch = idxs.shape[-1] // batch_size

        obs_all, acs_all, next_obs_all = self.dataloader.get_x_y_all()
        obs_all, acs_all, next_obs_all = obs_all.to(self.config.device), acs_all.to(self.config.device), next_obs_all.to(self.config.device)

        if self.config.agent.fit_input:
            if not self.config.agent.only_fit_first or (self.config.agent.only_fit_first and not dynamics_model.scaler.is_fit):
                fit_inputs = torch.cat([self.env.obs_preproc(obs_all), acs_all], dim=-1)
                dynamics_model.scaler.fit(fit_inputs)
        
        holdout_inputs = torch.cat([self.env.obs_preproc(obs_all[hold_idxs, :]), acs_all[hold_idxs, :]], dim=-1)
        holdout_labels = self.env.targ_proc(obs_all[hold_idxs, :], next_obs_all[hold_idxs, :])
        holdout_inputs = holdout_inputs.unsqueeze(0).expand(num_nets, -1, -1)
        holdout_labels = holdout_labels.unsqueeze(0).expand(num_nets, -1, -1)
        hold_num_batch = hold_idxs.shape[-1] // batch_size
        
        mes_loss_log = []
        hold_mes_loss_log = []
        for epoch in itertools.count():
            idxs = shuffle_rows(idxs)
            for batch_num in range(num_batch):
                
                batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]
                obs, acs, next_obs = obs_all[batch_idxs, :], acs_all[batch_idxs, :], next_obs_all[batch_idxs, :]
                
                loss = self.config.agent.dynamics_weight_decay_rate * dynamics_model.compute_decays()
                loss += 0.01 * (dynamics_model.max_logvar.sum() - dynamics_model.min_logvar.sum())
                
                x1 = torch.cat([self.env.obs_preproc(obs), acs], dim=-1)
                y1 = self.env.targ_proc(obs, next_obs)
                
                if self.config.agent.mc:
                    dynamics_model.sample_new_mask(batch_size=batch_size)
                dynamics_model.select_mask(batch_size)
                
                mean, logvar = dynamics_model(x1, ret_logvar=True)
                inv_var = torch.exp(-logvar)
                mes_loss = ((mean - y1) ** 2)
                if return_log:
                    mes_loss_log.append([i, mes_loss.mean(0).mean(0).detach().cpu().numpy()])
                mes_loss_sum = mes_loss.mean(-1).mean(-1).sum()
                train_losses = mes_loss * inv_var + logvar
                if self.config.agent.penalty:
                    train_losses += self.config.agent.penalty_rate/inv_var
                
                mes_loss_print = [mes_loss_sum.item()]
                
                if self.config.agent.fitting_error_correction:
                    fec_length = self.config.agent.fec_length

                    for l in range(fec_length-1):

                        next_vectorized = np.vectorize(self.dataloader.next_index)
                        next_idxs = next_vectorized(batch_idxs)
                        batch_idxs = next_idxs
                        # next_idxs = next_idxs[next_idxs>=0]
                        # new_batch_size = next_idxs.shape[-1]
                        
                        # predictions = self.env.obs_postproc(obs, mean)
                        signature = inspect.signature(self.env.obs_postproc)
                        if 'acs' in signature.parameters:
                            predictions = self.env.obs_postproc(obs, mean, acs)
                        else:
                            predictions = self.env.obs_postproc(obs, mean)
                        # predictions = predictions[next_idxs>=0, :]
                        predictions_preproc = self.env.obs_preproc(predictions)

                        obs, acs, next_obs = obs_all[next_idxs, :], acs_all[next_idxs, :], next_obs_all[next_idxs, :]
                
                        # idxs.apply_(self.dataloader.next_index)
                        inputs2 = torch.cat((predictions_preproc, acs), dim=-1)
                        if self.config.agent.mc:
                            dynamics_model.sample_new_mask(batch_size=batch_size)
                        dynamics_model.select_mask(batch_size)
                        mean2, logvar2 = dynamics_model(inputs2, ret_logvar=True)
                        if self.config.agent.fec_delta:
                            y2 = self.env.targ_proc(obs, next_obs)
                        else:
                            y2 = self.env.targ_proc(predictions, next_obs)
                        inv_var2 = torch.exp(-logvar2)
                        mes_loss2 = ((mean2 - y2) ** 2)
                        train_losses2 = mes_loss2 * inv_var2 + logvar2

                        if self.config.agent.penalty:
                            train_losses2 += self.config.agent.penalty_rate/inv_var2
                        
                        # train_losses2 = train_losses2.mean(-1)
                        # print('next_idxs.shape', next_idxs.shape)
                        # print('train_losses2.shape', train_losses2.shape)
                        next_idxs = torch.tensor(next_idxs, device=train_losses2.device)
                        # train_losses2[next_idxs<0] = 0
                        # train_losses2 = train_losses2.mean(-1).sum()
                        # next_idxs: num_nets, batch_size
                        
                        train_losses2[next_idxs<0, :] = torch.zeros((train_losses2.shape[-1]), device=train_losses2.device)
                        # clip_length = train_losses2.shape[0]
                        # train_losses2 = train_losses2.mean()
                        # train_losses2 = train_losses2*num_nets*(clip_length/(num_nets*batch_size))
                        
                        mes_loss_sum2 = mes_loss2.mean(-1)
                        mes_loss_sum2 = mes_loss_sum2[next_idxs>=0].mean()*num_nets

                        train_losses2 = train_losses2 * (self.config.agent.fec_decay ** (l+1))
                        # train_losses2 = train_losses2.mean(-1).mean(-1).sum()
                        if self.config.agent.fec_logsumexp:
                            train_losses = torch.logsumexp(torch.cat([train_losses.unsqueeze(0), train_losses2.unsqueeze(0)], dim=0), dim=0)
                            # ones_weights = torch.ones_like(train_losses.unsqueeze(0))
                            # train_losses = logsumexp(torch.cat([train_losses.unsqueeze(0), train_losses2.unsqueeze(0)], dim=0), dim=0, weights=torch.cat([ones_weights, ones_weights*(self.config.agent.fec_decay ** (l+1))], dim=0))
                        else:
                            # train_losses2 = train_losses2 * (self.config.agent.fec_decay ** (l+1))
                            train_losses += train_losses2
                        # loss += torch.log(train_losses2.exp() + loss.exp())
                        # train_losses += train_losses2
                        # train_losses = torch.logsumexp(torch.cat([train_losses.unsqueeze(0), train_losses2.unsqueeze(0)], dim=0), dim=0)
                        mean = mean2
                        mes_loss_print.append(mes_loss_sum2.item())
                
                train_losses = train_losses.mean(-1).mean(-1).sum()
                loss += train_losses
                loss += train_losses
                loss.backward()
                nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=20, norm_type=2)
                self.dynamics_optimizer.step()
                self.dynamics_optimizer.zero_grad()
                i += 1
                if i % 100 == 0:
                    print('epoch', self.exp_epoch, 'step', i, 'train_loss', loss.item(), 'MESLoss', mes_loss_print)
            if self.config.agent.fix_train_epoch:
                if epoch >= self.config.agent.train_epoch:
                    break
                else:
                    continue
            if self.config.agent.fix_train_batch:
                if i >= self.config.agent.train_batch:
                    break
                else:
                    continue
            with torch.no_grad():

                holdout_mse_losses = np.zeros((num_nets))
                holdout_mse_losses2 = np.zeros((holdout_labels.shape[-1]))
                for batch_num in range(hold_num_batch):
                    # num_nets, -1, -1
                    holdout_inputs_batch = holdout_inputs[:, batch_num * batch_size : (batch_num + 1) * batch_size, :]
                    holdout_labels_batch = holdout_labels[:, batch_num * batch_size : (batch_num + 1) * batch_size, :]

                    dynamics_model.select_mask(batch_size)
                    mean, logvar = dynamics_model(holdout_inputs_batch, ret_logvar=True)
                    
                    mes_loss = ((mean - holdout_labels_batch) ** 2)
                    holdout_mse_losses_batch = mes_loss.mean(-1).mean(-1)
                    holdout_mse_losses_batch = holdout_mse_losses_batch.detach().cpu().numpy()
                    holdout_mse_losses += holdout_mse_losses_batch

                    holdout_mse_losses2 += mes_loss.mean(0).mean(0).detach().cpu().numpy()

                holdout_mse_losses = holdout_mse_losses / hold_num_batch
                if return_log:
                    hold_mes_loss_log.append([i, holdout_mse_losses2 / hold_num_batch])
                if i % 100 == 0:
                    print('epoch', self.exp_epoch, 'step', i, 'test_loss', holdout_mse_losses.sum().item())
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                if self.config.agent.elite_size < self.config.agent.ensemble_size:
                    self.elite_model_idxes = sorted_loss_idx[:self.config.agent.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break

        num_nets = self.config.agent.elite_size
        net_particles = num_particles // num_nets
        self.model.select_mask(net_particles)
        self.model.set_elite_index(self.elite_model_idxes)
        
        self.controller.train_epoch(epoch_reward=epoch_reward)
        # for param in dynamics_model.parameters():
        #     param.requires_grad = False
        print('max_var', dynamics_model.max_logvar.exp().mean())
        print('min_var', dynamics_model.min_logvar.exp().mean())

        # 重制epoch lr
        if self.exp_epoch in self.config.agent.lr_scheduler:
            self.dynamics_scheduler.step()
            self.improvement_rate *= self.config.agent.dynamics_lr_gamma
        if return_log:
            return mes_loss_log, hold_mes_loss_log

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > self.improvement_rate:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False
        
    def sample(self, states, evaluation=False):
        """
        根据当前环境获取一个动作
        """
        # self.model.eval()
        action = self.controller.sample(states, self.exp_epoch, self.exp_step, evaluation)
        # print('action', action.max(), action.min())
        return action

    def add_data_hold(self, states, actions, indexs=[]):
        return
        assert states.shape[0] == actions.shape[0] + 1
        x = np.concatenate((self.env.obs_preproc(states[:-1]), actions), axis=1)
        y = self.env.targ_proc(states[:-1], states[1:])
        self.hold_dataloader.push(x, y)

    def add_data(self, states, actions, indexs=[], path_done=True):

        assert states.shape[0] == actions.shape[0] + 1

        obs = states[:-1]
        next_obs = states[1:]
        acs = actions
        y1 = self.env.targ_proc(obs, next_obs)
        print('add data: mean', y1.mean(axis=0), 'std', y1.std(axis=0))

        self.dataloader.push(obs, acs, next_obs, path_done=path_done)
    
    def add_data_old(self, states, actions, indexs=[]):

        assert states.shape[0] == actions.shape[0] + 1

        fec_length = self.config.agent.fec_length
        states_fec = []
        actions_fec = []
        states_fec.append(states[:-fec_length-1])
        for l in range(1, fec_length+1):
            actions_fec.append(actions[l-1:-(fec_length-l+1)])
            states_fec.append(states[l:-(fec_length-l+1)])

        self.dataloader.push(states_fec, actions_fec)

    def prediction(self, states, action, t=0, sample_epoch=0, print_info=False, return_reward_states=False, add_var=False, is_nopt=False, use_model=None, return_var = False, return_hidden = False):
        
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.config.device).float()
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, device=self.config.device).float()
        if(states.dim() == 1):
            states = states.unsqueeze(0).tile(self.config.agent.num_particles, 1).float()
        if(action.dim() == 1):
            action = action.unsqueeze(0).tile(states.shape[0], 1).float()
        
        proc_obs = self.env.obs_preproc(states)
        if is_nopt:
            expand_to_ts_format = self._expand_to_ts_format
            flatten_to_matrix = self._flatten_to_matrix
        else:
            expand_to_ts_format = self._expand_to_ts_format_dmbpo
            flatten_to_matrix = self._flatten_to_matrix_dmbpo

        proc_obs = expand_to_ts_format(proc_obs)
        action = expand_to_ts_format(action)

        inputs = torch.cat((proc_obs, action), dim=-1)
        # model = self.model
        if use_model is not None:
            model = use_model
        else:
            model = self.model
        # model = torch.compile(model)

        num_nets = self.config.agent.elite_size

        num_particles = self.config.agent.num_particles
        net_particles = num_particles // num_nets
        net_particles_batch = inputs.shape[1]
        batch_size = net_particles_batch // net_particles
        if self.config.agent.dropout_remask:
            if self.config.agent.mc:
                model.sample_new_mask(batch_size=net_particles)
            model.select_mask(net_particles)
        
        if model.batch_size != inputs.shape[1]:
            model.select_mask(inputs.shape[1])

        if self.config.agent.aleatoric == 'pets1':
            np.random.shuffle(self.elite_model_idxes)
            model.set_elite_index(self.elite_model_idxes)
        # mean, var = self.model(inputs)
        if model.lin0_w_e is None:
            model.set_elite_index(self.elite_model_idxes)
        if return_hidden:
            hidden = model.elite_forward(inputs, return_hidden=True)
            hidden = flatten_to_matrix(hidden)
            return hidden
        mean, var = model.elite_forward(inputs)

        if 'pets' in self.config.agent.aleatoric or add_var:
            predictions = mean + torch.randn_like(mean, device=self.config.device) * var.sqrt()
        elif return_reward_states:
            predictions = mean + torch.randn_like(mean, device=self.config.device) * var.sqrt()
            predictions_true = mean
        elif self.config.agent.aleatoric == 'test':
            predictions = mean + torch.randn_like(mean, device=self.config.device) * var.sqrt()
            mask = torch.rand(mean.shape, device=self.config.device) < 0.2  # 创建与输入张量形状相同的掩码张量，每个元素以概率p为True
            predictions = torch.where(mask, predictions, mean)  # 根据掩码张量选择元素值
        else:
            predictions = mean
        

        predictions = flatten_to_matrix(predictions)
        action = flatten_to_matrix(action)

        # prediction_obs = self.env.obs_postproc(states, predictions)

        signature = inspect.signature(self.env.obs_postproc)
        if 'acs' in signature.parameters:
            prediction_obs = self.env.obs_postproc(states, predictions, action)
        else:
            prediction_obs = self.env.obs_postproc(states, predictions)

        # prediction_obs = self.ts1(prediction_obs)
        if return_reward_states:
            predictions_true = flatten_to_matrix(predictions_true)
            if 'acs' in signature.parameters:
                prediction_obs_true = self.env.obs_postproc(states, predictions_true, action)
            else:
                prediction_obs_true = self.env.obs_postproc(states, predictions_true)
            
            return prediction_obs_true, prediction_obs
        if return_var:
            predictions_var = flatten_to_matrix(var)
            return prediction_obs, predictions_var
        return prediction_obs
    
    def _expand_to_ts_format_dmbpo(self, mat):
        dim = mat.shape[-1]
        num_nets = self.config.agent.elite_size
        reshaped = mat.view(num_nets, -1, dim)
        return reshaped
    
    def _flatten_to_matrix_dmbpo(self, ts_fmt_arr):

        dim = ts_fmt_arr.shape[-1]
        num_nets = self.config.agent.elite_size
        reshaped = ts_fmt_arr.view(-1, dim)
        return reshaped

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]
        num_nets = self.config.agent.elite_size
        reshaped = mat.view(-1, num_nets, self.config.agent.num_particles // num_nets, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(num_nets, -1, dim)
        return reshaped
    
    def _flatten_to_matrix(self, ts_fmt_arr):

        dim = ts_fmt_arr.shape[-1]
        num_nets = self.config.agent.elite_size
        reshaped = ts_fmt_arr.view(num_nets, -1, self.config.agent.num_particles // num_nets, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(-1, dim)
        return reshaped

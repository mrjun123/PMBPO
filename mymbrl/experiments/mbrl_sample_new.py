import mymbrl.agents as agents
import mymbrl.envs as envs
import numpy as np
import gym
import torch
import os
import time
import inspect
from mymbrl.utils import LogDict

class MBRLSampleNew:
    def __init__(self, config, writer):
        Env = envs.get_item(config.env)
        # signature = inspect.signature(Env)
        # if 'acs' in signature.parameters:
        if 'sim_dog_dmbpo' in config.env:
            env = Env(config)
            eval_env = Env(config)
        else:
            env = Env()
            eval_env = Env()
        self.eval_env = eval_env
        # if config.experiment.monitor:
        #     monitor_dir = os.path.join(config.run_dir, "monitor")
        #     env._max_episode_steps = config.experiment.horizon
        #     self.env = gym.wrappers.Monitor(env, monitor_dir, video_callable=lambda episode_id: True, force=True)
        # else:
        self.env = env
        self.writer = writer
        if hasattr(env, 'set_config'):
            env.set_config(config)
            eval_env.set_config(config)
        
        Agent = agents.get_item(config.agent.name)
        self.agent = Agent(config, env, writer)
        self.config = config
        
        self.env.seed(config.random_seed)
        self.eval_env.seed(config.random_seed)
        self.eval_epoch = -1000
        self.log_writer = LogDict(config.run_dir, 'run_data')

    def run(self):
        random_ntrain_iters = self.config.experiment.random_ntrain_iters
        ntrain_iters = self.config.experiment.ntrain_iters
        # def one_exp(self, epoch = 0, is_random = False, print_step=True):
        # self.agent.controller.reset()
        done = False
        cur_states = self.env.reset()
        pre_states = None
        pre_action = None
        self.agent.reset()
        actions = []
        rewards = []
        states = [cur_states]
        
        step = 0
        epoch_reward = 0
        all_step = 0

        # model_train_freq = self.config.agent.model_train_freq
        # random_horizon = self.config.experiment.random_horizon
        # horizon = self.config.experiment.horizon

        # train_epoch_num = horizon // model_train_freq
        # random_train_epoch_num = random_horizon // model_train_freq
        e_num = 0
        
        for i in range(random_ntrain_iters + ntrain_iters):
            epoch = i - random_ntrain_iters
            is_random = (epoch < 0)
            # epoch = epoch + 1
            self.agent.set_epoch(epoch)
            horizon = self.config.experiment.horizon
            if is_random:
                horizon = self.config.experiment.random_horizon
            for epoch_step in range(horizon):
                # if self.config.experiment.monitor:
                #     self.env.render()
                self.agent.set_step(epoch_step)
                if hasattr(self.env, 'set_step'):
                    self.env.set_step(epoch_step)
                path_done = (done or step >= horizon)
                all_step += 1

                start_time = time.time()

                need_train = (not is_random and all_step % self.config.agent.model_train_freq == 0)
                need_train = (all_step % self.config.agent.model_train_freq == 0)
                if not self.config.experiment.random_train:
                    need_train = (not is_random and need_train)
                # need_train = ((epoch_step + 1) % self.config.agent.model_train_freq == 0)
                
                # need_train = (epoch > 0 and all_step % self.config.agent.model_train_freq == 0)
                if path_done or need_train:
                    # 若结束则默认该动作对状态没有变化
                    # states[-1] = cur_states
                    if len(actions) > 0:
                        states, actions = tuple(map(lambda l: np.stack(l, axis=0),
                                                    (states, actions)))
                        self.agent.add_data(states, actions, path_done=path_done)
                    
                    if need_train:
                        if epoch > 0 and self.config.experiment.evaluation:
                            self.evaluation(epoch)
                        self.agent.train(epoch_reward=epoch_reward)
                        epoch_reward = 0
                    if path_done:
                        cur_states = self.env.reset()
                        pre_states = None
                        self.agent.reset()
                        step = 0
                        e_num += 1
                    actions = []
                    states = [cur_states]
                
                if is_random and epoch_step % self.config.agent.model_train_freq == 0:
                    epoch_reward = 0
                action = None
                if is_random:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.sample(cur_states)
                
                end_time = time.time()
                step_time = end_time - start_time
                if epoch_step != 0 and not is_random:
                    self.log_writer.log(f'step-time-epoch-{epoch}', step_time)
                # run_times.append(step_time)
                self.log_writer.log(f'step-time-enum-{e_num}', step_time)
                for i in range(self.config.experiment.step_num):
                    next_state, reward, done, info= self.env.step(action)
                
                epoch_reward += reward
                if self.config.experiment.noise > 0:
                    next_state = np.array(next_state) + np.random.uniform(
                        low=-self.config.experiment.noise, high=self.config.experiment.noise, size=next_state.shape
                    )
                if not is_random:
                    print("action_step:", epoch_step, "action_reward:", reward)
                
                if self.config.experiment.record_each_epoch:
                    self.writer.add_scalar(f'mbrl/rewards/epoch{epoch}', reward, epoch_step)
                
                terminal = done
                if "TimeLimit.truncated" in info and info["TimeLimit.truncated"]:
                    terminal = False

                self.agent.add_data_step(cur_states, action, reward, next_state, terminal)
                is_start = False
                if pre_states is None:
                    is_start = True
                self.agent.controller.add_two_step_data(pre_states, pre_action, cur_states, action, reward, next_state, terminal, is_start)

                actions.append(action)
                states.append(next_state)
                rewards.append(reward)
                pre_states = cur_states
                pre_action = action
                cur_states = next_state
                step += 1
                
            rewards = np.array(rewards)
            self.writer.add_scalar('mbrl/rewards', rewards.sum(), epoch)
            rewards = []
            self.log_writer.save()
        self.log_writer.save()
            
    def evaluation(self, epoch):
        if epoch == self.eval_epoch:
            return
        self.eval_epoch = epoch
        test_step = 0
        sum_reward = 0
        cur_states = self.eval_env.reset()
        done = False
        actions = []
        states = [cur_states]
        while (not done) and (test_step != self.config.experiment.horizon):
            action = self.agent.sample(cur_states, evaluation=True)
            next_state, reward, done, info= self.eval_env.step(action)
            cur_states = next_state
            sum_reward += reward
            test_step += 1
            actions.append(action)
            states.append(next_state)
        states, actions = tuple(map(lambda l: np.stack(l, axis=0), (states, actions)))
        # self.agent.add_data_hold(states, actions)
        if epoch % 5 == 0 and self.config.experiment.plt_info:
            self.plt_long_pred(states, actions, epoch)
        if self.config.experiment.save_controller:
            self.agent.controller.save_model()
        
        self.writer.add_scalar('mbrl/evaluation/rewards', sum_reward, epoch)

    def plt_long_pred(self, states, actions, epoch):
        states = np.array(states)
        actions = np.array(actions)
        # states: 26, actions: 25
        max_predict_length = 5
        states_len = states.shape[0]
        self.log_writer.log(f'true-states-epoch-{epoch}', states)
        self.log_writer.log(f'true-actions-epoch-{epoch}', actions)
        
        if states_len < 2:
            return
        # 数据开始位置
        for l in range(states_len):
            start_status = states[l]
            predict_length = max_predict_length
            if states_len - 1 - l < predict_length:
                predict_length = states_len - l - 1
            if predict_length <= 0:
                break

            cur_obs = start_status
            ob_dim = cur_obs.shape[-1]
            cur_obs = cur_obs.reshape(1, ob_dim)
            cur_obs = np.tile(cur_obs, (self.config.agent.num_particles, 1))
            
            obs_log = []
            acs_log = []
            obs_log.append(cur_obs)
            
            for t in range(predict_length):
                cur_acs = actions[l+t]
                with torch.no_grad():
                    next_obs = self.agent.prediction(cur_obs, cur_acs, t)
                cur_obs = next_obs
                obs_log.append(cur_obs.detach().clone().cpu().numpy())
                acs_log.append(cur_acs)
            
            # obs_log = np.array(obs_log)
            # acs_log = np.array(acs_log)
            
            self.log_writer.log(f'states-epoch-{epoch}', obs_log)
            self.log_writer.log(f'actions-epoch-{epoch}', acs_log)
        

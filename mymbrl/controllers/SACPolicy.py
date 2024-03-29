import mymbrl.optimizers as optimizers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mymbrl.utils import swish
from .controller import Controller
from mymbrl.mfrl.sac.replay_memory import ReplayMemory
from mymbrl.mfrl.sac.sac import SAC
from mymbrl.envs.utils import termination_fn
import copy
from numbers import Number
import time

"""
使用Agent训练的Model和收集的数据训练一个SAC Agent。
"""
class SACPolicy(Controller):
    def __init__(self, agent, is_torch=True, writer=None):
        
        super(SACPolicy, self).__init__(agent, is_torch, writer)
        env = self.env
        config = self.config
        self.config.agent.SACPolicy.epoch_length = self.config.experiment.horizon
        args = self.config.agent.SACPolicy
        
        self.dO, self.dU = env.observation_space.shape[0], env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low
        self.action_dim = self.ac_ub.shape[0]
        
        actions_num = config.agent.predict_length
        self.actions_num = actions_num

        self.lower_bound = torch.tile(torch.tensor(self.ac_lb), [actions_num]).to(self.config.device)
        self.upper_bound = torch.tile(torch.tensor(self.ac_ub), [actions_num]).to(self.config.device)
        
        self.init_sol = (self.lower_bound + self.upper_bound) / 2

        self.predict_env = PredictEnv(env, agent, config)
        # epoch_length = config.experiment.horizon
        # model_retain_epochs = args.model_retain_epochs
        # rollouts_per_epoch = args.rollout_batch_size * epoch_length / config.agent.model_train_freq
        # model_steps_per_epoch = int(1 * rollouts_per_epoch)

        self.rollout_length = 1
        self.train_policy_steps = 0

        # new_pool_size = model_retain_epochs * model_steps_per_epoch
        # self.model_pool = ReplayMemory(new_pool_size)
        self.model_pool = self.resize_model_pool(self.rollout_length)
        self.env_pool = ReplayMemory(args.replay_size)
        self.SAC_agent = SAC(env.MODEL_IN - env.action_space.shape[0], env.action_space, args)
        # self.SAC_agent = SAC(env.observation_space.shape[0], env.action_space, args)
        if args.mpc_rollout:
            Optimizer = optimizers.get_item(config.agent.optimizer)
            self.optimizer = Optimizer(
                sol_dim=actions_num * self.action_dim,
                config=self.config,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound
            )
        self.best_epoch_reward = -1e5
        self.best_SAC_agent = None
        self.best_start_SAC_agent = None

    def reset(self):
        self.train_policy_steps = 0
        pass
        
    def sample(self, states, epoch=-1, step=-1, evaluation=False):
        if not evaluation:
            self.train_policy_steps += self.train_step()
            
        # 使用SAC训练完成的Policy
        with torch.no_grad():
            # states = torch.from_numpy(states).to(self.config.device).float()
            states = self.env.obs_preproc(states)
            action = self.SAC_agent.select_action(states)
            # action = torch.flatten(self.policy_model(states))  # Ensure ndims=1
        # action = action.data.cpu().numpy()
        
        return action
    
    def add_data_step(self, cur_state, action, reward, next_state, done):
        self.env_pool.push(cur_state, action, reward, next_state, done)

    def train_epoch(self, epoch_reward=0):
        
        args = self.config.agent.SACPolicy
        train_num = args.epoch_train_num

        if args.mpc_rollout and self.exp_epoch >= args.mpc_rollout_min_epoch:
            self.mpc_rollout_model()
        test_params = self.config.test_params
        if self.config.test and test_params.policy_fallback:
            
            print('epoch_reward', epoch_reward, 'best_epoch_reward', self.best_epoch_reward)
            if (epoch_reward + test_params.best_policy_margin <= self.best_epoch_reward) and self.exp_epoch > 2:
                # self.SAC_agent = copy.deepcopy(self.best_SAC_agent)
                self.SAC_agent.load_dict(self.best_SAC_agent)
                # 若出现回退，则使用最新的真实数据再训练一定步数
                fallback_train_num = test_params.fallback_train_num
                if fallback_train_num:
                    for o in range(fallback_train_num):
                        self.train_step()
                print('load best agent')
            elif epoch_reward >= self.best_epoch_reward:
                self.best_SAC_agent = self.SAC_agent.get_dict()
                self.best_epoch_reward = epoch_reward
            else:
                self.best_SAC_agent = self.SAC_agent.get_dict()
                pass
            if test_params.model_train_close_aet:
                # self.SAC_agent.automatic_entropy_tuning = False
                self.SAC_agent.set_automatic_entropy_tuning(False)
        
        if self.exp_epoch >= args.epoch_train_min_epoch:
            for i in range(train_num):
                self.rollout_model(args.rollout_batch_size // train_num)
                if args.epoch_num_train_repeat > 0:
                    self.train_step(is_epoch=(i+1))
        if self.config.test and test_params.policy_fallback:
            if test_params.model_train_close_aet:
                self.SAC_agent.set_automatic_entropy_tuning(True)
            # self.SAC_agent.automatic_entropy_tuning = True
        # if args.epoch_num_train_repeat > 0:
        #     self.train_step(is_epoch=True)
        # if self.config.test and self.exp_epoch >= 0:
        #     for n in range(5):
        #         self.rollout_model()
        #         for t in range(100):
        #             self.train_step()
        # else:
        #     self.rollout_model()
        
    def rollout_model(self, rollout_batch_size):
        if self.exp_epoch < 0:
            return 0
            
        total_step = self.exp_epoch*self.config.experiment.horizon + self.exp_step
        # 生成训练数据
        if total_step <= 0:
            return
        env_pool = self.env_pool
        args = self.config.agent.SACPolicy
        
        epoch_step = self.exp_epoch

        new_rollout_length = self.set_rollout_length(epoch_step)
        if self.rollout_length != new_rollout_length:
            self.rollout_length = new_rollout_length
            self.model_pool = self.resize_model_pool(self.rollout_length, self.model_pool)
        
        state, action, reward, next_state, done = env_pool.sample_all_batch(rollout_batch_size)
        # num_particles = self.config.agent.num_particles
        # num_example = state.shape[0]
        # spare_len = num_example % num_particles
        # new_num_example = num_example - spare_len
        # state = state[:new_num_example]
        state_reward = state
        pre_nonterm_mask = (np.zeros(state.shape[0]) == 0)

        for i in range(self.rollout_length):
            # TODO: Get a batch of actions
            pre_state = self.env.obs_preproc(state)
            action = self.SAC_agent.select_action(pre_state)
            next_states, next_obs_reward, rewards, terminals, _ = self.predict_env.step(state, action, i, self.rollout_length)
            # TODO: Push a batch of samples
            self.model_pool.push_batch([(state_reward[j], action[j], rewards[j], next_obs_reward[j], terminals[j]) for j in range(state.shape[0]) if pre_nonterm_mask[j]])
            nonterm_mask = ~terminals.squeeze(-1)
            if nonterm_mask.sum() == 0:
                break
            state = next_states
            state_reward = next_obs_reward
            pre_nonterm_mask[nonterm_mask == False] = False

    def mpc_rollout_model(self):
        predict_length = self.config.agent.predict_length
        num_particles = self.config.agent.num_particles
        action_dim = self.dU
        mpc_rollout_batch_size = self.config.agent.SACPolicy.mpc_rollout_batch_size

        env_state, _, _, _, _ = self.env_pool.sample(int(mpc_rollout_batch_size))
        for i in range(env_state.shape[0]):
            states = env_state[i]
            def states_cost_func(ac_seqs, return_torch = True, sample_epoch = -1, solution = False):
                if not isinstance(ac_seqs, torch.Tensor):
                    ac_seqs = torch.tensor(ac_seqs, device=self.config.device).float()

                batch_size = ac_seqs.shape[0]
                ac_seqs = ac_seqs.reshape(batch_size, predict_length, 1, action_dim)
                ac_seqs = ac_seqs.transpose(0, 1).contiguous()
                ac_seqs = ac_seqs.expand(-1, -1, num_particles, -1)

                return self.mpc_cost_fun(ac_seqs, states, return_torch, sample_epoch=sample_epoch, solution=solution)
            
            opt_action_next = self.optimizer.obtain_solution(
                states_cost_func, 
                self.init_sol
            )
            last_cost = states_cost_func(opt_action_next[None], return_torch = False, solution=True).mean().item()
            print('step', i, 'last_cost', last_cost)
            mpc_obs = np.array(self.mpc_obs)
            mpc_acs = np.array(self.mpc_acs)
            mpc_nonterm_masks = np.array(self.mpc_nonterm_masks)
            # mpc_nonterm_masks = self.mpc_nonterm_masks

            # prediction_length, particle, dim
            obs = mpc_obs[:-1, :, :]
            next_obs = mpc_obs[1:, :, :]

            obs = obs.reshape(-1, obs.shape[-1])
            next_obs = next_obs.reshape(-1, next_obs.shape[-1])
            acs = mpc_acs.reshape(-1, mpc_acs.shape[-1])
            mpc_nonterm_masks = mpc_nonterm_masks.reshape(-1, 1)
            # acs = mpc_acs

            costs = self.env.obs_cost_fn_cost(next_obs) + self.env.ac_cost_fn_cost(acs)
            costs = costs[:, None]
            rewards = -costs
            terminals = termination_fn(self.config.env, next_obs, acs, rewards)
            # mpc_nonterm_masks = self.mpc_nonterm_masks
            
            self.model_pool.push_batch([(obs[l], acs[l], rewards[l], next_obs[l], terminals[l]) for l in range(obs.shape[0]) if mpc_nonterm_masks[l]])

    def save_model(self):
        self.SAC_agent.save_model(self.config.run_dir, epoch=self.exp_epoch)
    
    # 每一步训练策略
    def train_step(self, is_epoch=False):

        if self.exp_epoch < 0:
            return 0
        
        args = self.config.agent.SACPolicy
        model_pool = self.model_pool
        env_pool = self.env_pool
        agent = self.SAC_agent
        total_step = self.exp_epoch * self.config.experiment.horizon + self.exp_step
        train_step = self.train_policy_steps
        # print('train_step', train_step, 'total_step', total_step)
        if args.train_every_n_steps > 0 and total_step % args.train_every_n_steps > 0:
            return 0
        
        if len(env_pool) <= args.min_pool_size:
            return 0
        
        max_train_repeat_per_step = max(args.max_train_repeat_per_step, args.num_train_repeat*args.num_train_data_repeat*args.num_train_env_data_repeat)
        if train_step > max_train_repeat_per_step * total_step:
            return 0
        num_train_repeat = args.num_train_repeat
        if is_epoch:
            num_train_repeat = args.epoch_num_train_repeat
            print('policy train epoch', is_epoch, 'success')

        for i in range(num_train_repeat):
            real_ratio = args.real_ratio
            if not is_epoch and self.config.test and isinstance(self.config.test_params.step_real_ratio, Number):
                real_ratio = self.config.test_params.step_real_ratio
            env_batch_size = int(args.policy_train_batch_size * real_ratio)
            model_batch_size = args.policy_train_batch_size - env_batch_size

            env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))
            env_state = self.env.obs_preproc(env_state)
            env_next_state = self.env.obs_preproc(env_next_state)
            # 有模型数据则使用真实数据和模型数据
            for j in range(args.num_train_env_data_repeat):
                if model_batch_size > 0 and len(model_pool) > 0:
                    model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
                    
                    model_state = self.env.obs_preproc(model_state)
                    model_next_state = self.env.obs_preproc(model_next_state)
                    
                    batch_state = np.concatenate((env_state, model_state), axis=0)
                    batch_action = np.concatenate((env_action, model_action), axis=0)
                    batch_reward = np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0)
                    batch_next_state = np.concatenate((env_next_state, model_next_state), axis=0)
                    batch_done = np.concatenate( (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
                else:
                    # 无模型数据使用真实数据
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

                batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
                batch_done = (~batch_done).astype(int)
                # 策略梯度
                # for j in range(args.num_train_data_repeat):
                for k in range(args.num_train_data_repeat):
                    start_time = time.time()
                    agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    # print('elapsed_time', elapsed_time)
                    

        return args.num_train_repeat*args.num_train_env_data_repeat*args.num_train_data_repeat

    def set_rollout_length(self, epoch_step):
        args = self.config.agent.SACPolicy
        rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                                / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                                args.rollout_min_length), args.rollout_max_length))
        return int(rollout_length)
    
    def resize_model_pool(self, rollout_length, model_pool = None):
        args = self.config.agent.SACPolicy
        model_train_freq = self.config.agent.model_train_freq
        rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / model_train_freq
        model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
        new_pool_size = args.model_retain_epochs * model_steps_per_epoch
        if args.mpc_rollout and self.exp_epoch >= args.mpc_rollout_min_epoch:
            predict_length = self.config.agent.predict_length
            num_particles = self.config.agent.num_particles
            mpc_rollout_batch_size = args.mpc_rollout_batch_size
            new_pool_size += predict_length * num_particles * mpc_rollout_batch_size * int(args.epoch_length / model_train_freq)
        new_model_pool = ReplayMemory(new_pool_size)

        if model_pool is not None:
            sample_all = model_pool.return_all()
            new_model_pool.push_batch(sample_all)

        return new_model_pool
    
class PredictEnv:
    def __init__(self, env, agent, config):
        self.agent = agent
        self.env = env
        self.config = config
        self.env_name = config.env
        
    def step(self, obs, act, step=-1, rollout_length=-2):
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
        with torch.no_grad():
            if self.config.agent.aleatoric == 'dmbpo':
                # add_var
                next_obs, next_obs_reward = self.agent.prediction(obs, act, return_reward_states=True)
                next_obs = next_obs.cpu().numpy()
                next_obs_reward = next_obs_reward.cpu().numpy()
            elif self.config.agent.aleatoric == 'dmbpo_test':
                if step == rollout_length - 1:
                    next_obs = self.agent.prediction(obs, act, add_var=True)
                else:
                    next_obs = self.agent.prediction(obs, act)
                next_obs = next_obs.cpu().numpy()
                next_obs_reward = next_obs
            else:
                next_obs = self.agent.prediction(obs, act)
                next_obs = next_obs.cpu().numpy()
                next_obs_reward = next_obs
        
        costs = self.env.obs_cost_fn_cost(next_obs_reward) + self.env.ac_cost_fn_cost(act)
        costs = costs[:, None]
        rewards = -costs
        terminals = termination_fn(self.env_name, obs, act, next_obs_reward)
        return next_obs, next_obs_reward, rewards, terminals, {}
    
    def _get_logprob(self, x, means, variances):
        k = x.shape[-1]
        ## [ num_networks, batch_size ]
        log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)
        ## [ batch_size ]
        log_prob = np.log(prob)
        stds = np.std(means, 0).mean(-1)
        return log_prob, stds

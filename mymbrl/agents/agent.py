class Agent:
    def __init__(self):
        pass

    def train(self):
        """训练一个agent
        """
        pass
    
    def sample(self):
        """根据当前环境获取一个动作
        """
        pass
    def set_model(self, model):
        self.model = model
        
    def reset(self):
        self.controller.reset()
        
    def add_data(self, states, actions, indexs=[]):
        pass
    
    def set_epoch(self, exp_epoch):
        self.exp_epoch = exp_epoch
        self.controller.set_epoch(exp_epoch)
    
    def set_step(self, exp_step):
        self.exp_step = exp_step
        self.controller.set_step(exp_step)
    
    def add_data_step(self, cur_state, action, reward, next_state, done):
        self.controller.add_data_step(cur_state, action, reward, next_state, done)
    
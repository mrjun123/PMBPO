import numpy as np
from scipy.spatial.transform import Rotation as R

def termination_fn(env_name, obs, act, next_obs):
        if env_name == 'half_cheetah':
            done = (np.zeros(obs.shape[0]) == 1)
            done = done[:, None]
            return done
        else:
            done = (np.zeros(obs.shape[0]) == 1)
            done = done[:, None]
            return done
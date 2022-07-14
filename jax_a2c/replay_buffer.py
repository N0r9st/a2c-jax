from abc import ABC, abstractmethod
import pickle
import numpy as np
import collections
from jax_a2c.utils import Experience

class BaseBuffer(ABC):
    @abstractmethod
    def __init__(self, size:int, *args, **kwargs):
        pass
    @abstractmethod
    def add_experience(self, experience):
        raise NotImplementedError

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @abstractmethod
    def get_batch(self):
        raise NotImplementedError


class ReplayBuffer(BaseBuffer):
    def __init__(self, size:int, o_size, a_size, b_size, obs_rms, ret_rms, *args, **kwargs):

        self.size = size
        self.b_size = b_size
        self.observations = np.zeros((size, o_size))
        self.actions = np.zeros((size, a_size))
        self.next_observations = np.zeros((size, o_size))
        self.rewards = np.zeros(size)
        self.dones = np.zeros(size)
        self.states = [None] * self.size
        self._count = 0

        self.obs_rms = obs_rms
        self.ret_rms = ret_rms
        self.epsilon = 1e-8
        self.clip_obs = 10
        self.clip_reward = 10

    def add_experience(self, experience: Experience):
        num_steps = experience.observations.shape[0]
        in_observations = self.concat_first_two_dims(experience.observations)
        in_next_observations = self.concat_first_two_dims(experience.next_observations)
        in_actions = self.concat_first_two_dims(experience.actions)
        in_dones = self.concat_first_two_dims(experience.dones[-num_steps:])
        in_rewards = self.concat_first_two_dims(experience.rewards)

        in_states = self.concat_first_two_dims_list(experience.states[:num_steps])


        current_size = in_observations.shape[0]
        start = self._count % self.size        
        end = start + current_size
        print('--------------------------------')
        print(start, end)

        self._count += current_size

        if end > self.size:
            split = self.size - start
            end = end % self.size
            print(split)
            print(start, '->', '->', end)
            self.observations[start:] = in_observations[:split]
            self.actions[start:] = in_actions[:split]
            self.next_observations[start:] = in_next_observations[:split]
            self.dones[start:] = in_dones[:split]
            self.states[start:] = in_states[:split]
            self.rewards[start:] = in_rewards[:split]

            self.observations[:end] = in_observations[split:]
            self.actions[:end] = in_actions[split:]
            self.next_observations[:end] = in_next_observations[split:]
            self.dones[:end] = in_dones[split:]
            self.states[:end] = in_states[split:]
            self.rewards[:end] = in_rewards[split:]
            return 

        self.observations[start:end] = in_observations
        self.actions[start:end] = in_actions
        self.next_observations[start:end] = in_next_observations
        self.dones[start:end] = in_dones
        self.rewards[start:end] = in_rewards
        self.states[start:end] = in_states
        
        

        print('--------------------------------')

    @staticmethod
    def concat_first_two_dims(x):
        return x.reshape(
            (x.shape[0]*x.shape[1], ) + x.shape[2:]
        )
    @staticmethod
    def concat_first_two_dims_list(x):
        """ LISTS: (num_steps, num_envs, obj) -> (num_steps*num_envs, obj)
        """
        out = []
        for step_slice in x:
            out += (step_slice)
        return out 

    def get_batch(self, b_size=None, normalize=True):
        if b_size is None:
            b_size = self.b_size
            
        indices = np.random.choice(min(self.size, self._count), size=b_size, replace=self.size < b_size)
        
        observations = self.observations[indices]
        rewards = self.rewards[indices]

        if normalize:
            observations = np.clip(
                (observations- self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
            rewards = np.clip(rewards / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        return  Experience(
            states=[self.states[i] for i in indices],
            observations=observations,
            actions=self.actions[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
            rewards=rewards,
            values=None,
        )


        
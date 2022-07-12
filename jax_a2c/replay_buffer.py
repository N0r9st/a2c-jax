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
    def __init__(self, size:int, o_size, a_size, b_size, *args, **kwargs):

        self.size = size
        self.b_size = b_size
        self.observations = np.zeros((size, o_size))
        self.actions = np.zeros((size, a_size))
        self.next_observations = np.zeros((size, o_size))
        self.rewards = np.zeros(size)
        self.dones = np.zeros(size)
        self.states = [None] * self.size
        self._count = 0

    def add_experience(self, experience:Experience):
        
        current_size = len(experience.observations)
        start = self._count % self.size        
        end = start + current_size

        if end > self.size:
            split = self.size - start
            end = end % self.size

            self.observations[start:] = experience.observations[:split]
            self.actions[start:] = experience.actions[:split]
            self.next_observations[start:] = experience.next_observations[:split]
            self.dones[start:] = experience.dones[:split]
            self.states[start:] = experience.states[:split]

            self.observations[:end] = experience.observations[split:]
            self.actions[:end] = experience.actions[split:]
            self.next_observations[:end] = experience.next_observations[split:]
            self.dones[:end] = experience.dones[split:]
            self.states[:end] = experience.states[split:]
            return 

        self.observations[start:end] = experience.observations
        self.actions[start:end] = experience.actions
        self.next_observations[start:end] = experience.next_observations
        self.dones[start:end] = experience.dones
        self.rewards[start:end] = experience.rewards
        self.states[start:end] = experience.states

    def get_batch(self, b_size=None):
        if b_size is None:
            b_size = self.b_size
            
        indices = np.random.choice(self.size, size=b_size, replace=self.size < b_size)

        return Experience(
            states=self.states[indices],
            observations=self.observations[indices],
            actions=self.actions[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
            rewards=self.rewards[indices],
            values=None
        )
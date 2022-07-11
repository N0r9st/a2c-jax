


from abc import ABC, abstractmethod
import jax
from flax import linen as nn
from jax_a2c.a2c import p_step

class BaseTrainer(ABC):
    @abstractmethod
    def train_q(self, data: dict, params: dict):
        pass

    @abstractmethod
    def train_p(self, data: dict, params: dict):
        pass

    def end_epoch(self):
        pass

class BaseCollector(ABC):
    @abstractmethod
    def collect_experience(self, params):
        pass


class Trainer(BaseTrainer):
    def __init__(self, state, prngkey):
        
        self.state = state
        self.prngkey = prngkey

    def train_p(self, data: dict, args: dict,) -> dict:
        self.tate, (loss, loss_dict) = p_step(
            self.state, 
            data,
            self.prngkey,
            constant_params=args['train_constants'],
            )

        return loss_dict

    def train_q(self, data, args): 
        pass


class BaseA2CAlgorithm(ABC):
    @abstractmethod
    def collect_experience(self,):
        pass

    @abstractmethod
    def p_step(self):
        pass

    @abstractmethod
    def q_step(self):
        pass

    @abstractmethod
    def end_epoch(self):
        pass
    

class BaseCollector(ABC):
    @abstractmethod
    def collect_path(self):
        pass


class StandartCollector(ABC):
    def __init__(self, env_params) -> None:
        pass
        
    

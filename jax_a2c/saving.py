from ctypes import Union
from typing import Optional, Tuple
from flax.training.train_state import TrainState
import flax
import os
import pickle
import jax
import jax.numpy as jnp
from flax.core import freeze
from copy import deepcopy

from optax._src.base import EmptyState
from optax._src.transform import ScaleByAdamState, ScaleByScheduleState


def save_state(path: str, state: TrainState, additional: dict):
    # pickling state dictionary. 
    # State has all information about current train state: weights, optimizer stats and current step
    state_dict = flax.serialization.to_state_dict(state)
    with open(path, 'wb') as handle:
        pickle.dump((additional, state_dict), handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_state(path: str, state: TrainState) -> Tuple[TrainState, dict]:
    """
    loading state from pickled dictionary. In python 3.7 (and below?) use pickle5 library
    """

    with open(path, 'rb') as handle:
        additional, state_dict = pickle.load(handle)
    state_dict = jax.tree_map(jnp.array, state_dict)
    state_dict['step'] = state_dict['step'].item()
    state = flax.serialization.from_state_dict(state, state_dict)
    
    return state, additional

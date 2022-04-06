import functools
from typing import Any, Callable, Dict, Tuple
from collections import namedtuple
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

import jax_a2c.env_utils

Array = Any
PRNGKey = Any
ModelClass = Any
Experience = namedtuple(
    'Experience', 
    ['observations', 'actions', 'rewards', 'values', 'dones', 'states'])

def create_train_state(
    prngkey: PRNGKey, 
    model: ModelClass,
    envs: jax_a2c.env_utils.SubprocVecEnv,
    learning_rate: float, 
    decaying_lr: bool, 
    max_norm: float,
    decay: float,
    eps: float,
    train_steps: int = 0) -> TrainState:

    dummy_input = envs.reset()
    variables = model.init(prngkey, dummy_input)
    params = variables['params']

    if decaying_lr:
        lr = optax.linear_schedule(
            init_value = learning_rate, end_value=0.,
            transition_steps=train_steps)
    else:
        lr = learning_rate

    tx = optax.chain(
        optax.clip_by_global_norm(max_norm),
        optax.rmsprop(learning_rate=lr, decay=decay, eps=eps)
        )
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx)

    return state

class QTrainState(TrainState):
    q_fn: Callable  = struct.field(pytree_node=False)
    

def create_train_state(
    prngkey: PRNGKey, 
    policy_model: ModelClass,
    qf_model: ModelClass,
    envs: jax_a2c.env_utils.SubprocVecEnv,
    learning_rate: float, 
    decaying_lr: bool, 
    max_norm: float,
    decay: float,
    eps: float,
    train_steps: int = 0) -> QTrainState:

    dummy_input = envs.reset()
    dummy_action = np.stack([envs.action_space.sample() for _ in range(envs.num_envs)])

    policy_variables = policy_model.init(prngkey, dummy_input)
    policy_params = policy_variables['params']

    prngkey, _ = jax.random.split(prngkey)
    qf_variables = qf_model.init(prngkey, dummy_input, dummy_action)
    qf_params = qf_variables['params']

    if decaying_lr:
        lr = optax.linear_schedule(
            init_value = learning_rate, end_value=0.,
            transition_steps=train_steps)
    else:
        lr = learning_rate

    tx = optax.chain(
        optax.clip_by_global_norm(max_norm),
        optax.rmsprop(learning_rate=lr, decay=decay, eps=eps)
        )
    state = QTrainState.create(
        apply_fn=policy_model.apply,
        params={'policy_params': policy_params, 'qf_params': qf_params},
        q_fn=qf_model.apply,
        tx=tx)

    return state

@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(
        rewards: np.ndarray,
        terminal_masks: np.ndarray,
        values: np.ndarray,
        discount: float,
        gae_param: float):
    assert rewards.shape[0] + 1 == values.shape[0]
    advantages = []
    gae = 0.
    for t in reversed(range(len(rewards))):
        value_diff = discount * values[t + 1] * terminal_masks[t + 1] - values[t]
        delta = rewards[t] + value_diff
        gae = delta + discount * gae_param * terminal_masks[t + 1] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)

def collect_experience(
    prngkey: PRNGKey,
    next_obs_and_dones: Array,
    envs: jax_a2c.env_utils.SubprocVecEnv, 
    num_steps: int, 
    policy_fn: Callable, 
    )-> Tuple[Array, ...]:

    envs.training = True

    next_observations, dones = next_obs_and_dones

    observations_list = []
    actions_list = []
    rewards_list = []
    values_list = []
    dones_list = [dones]
    states_list = [envs.get_state()]

    for _ in range(num_steps):
        observations = next_observations
        _, prngkey = jax.random.split(prngkey)
        values, actions = policy_fn(prngkey, observations) 
        next_observations, rewards, dones, info = envs.step(actions)
        observations_list.append(observations)
        actions_list.append(np.array(actions))
        rewards_list.append(rewards)
        values_list.append(values[..., 0])
        dones_list.append(dones)
        states_list.append(envs.get_state())
        

    _, prngkey = jax.random.split(prngkey)
    values, actions = policy_fn(prngkey, next_observations) 
    values_list.append(values[..., 0])

    experience = Experience(
        observations=np.stack(observations_list),
        actions=np.stack(actions_list),
        rewards=np.stack(rewards_list),
        values=np.stack(values_list),
        dones=np.stack(dones_list),
        states=states_list,
    )
    return (next_observations, dones), experience

@functools.partial(jax.jit, static_argnums=(1,2))
def process_experience(
    experience: Tuple[Array, ...], 
    gamma: float = .99, 
    lambda_: float = .95,
    ):
    # observations, actions, rewards, values, dones = experience
    observations = experience.observations
    actions = experience.actions
    rewards = experience.rewards
    values = experience.values
    dones = experience.dones

    dones = jnp.logical_not(dones).astype(float)
    advantages = gae_advantages(rewards, dones, values, gamma, lambda_)
    returns = advantages + values[:-1]
    trajectories = (observations, actions, returns, advantages)
    num_agents, actor_steps = observations.shape[:2]
    trajectory_len = num_agents * actor_steps
    trajectories = tuple(map(
        lambda x: np.reshape(x, (trajectory_len,) + x.shape[2:]), trajectories))
    return trajectories

@functools.partial(jax.jit, static_argnums=(3, 4))
def process_rewards(dones, rewards, bootstrapped_values, gamma):
    masks = jnp.cumprod((1-dones)*gamma, axis=0)/gamma
    k_returns = (rewards*masks[:-1]).sum(axis=0) + bootstrapped_values * masks[-1]
    return k_returns


@functools.partial(jax.jit, static_argnums=(3,))
def get_mc_returns(rewards, dones, last_values, gamma):
    masks = 1-dones
    returns = jnp.zeros_like(rewards)
    returns = returns.at[-1].set(last_values)
    for i, rews in reversed(list(enumerate(rewards))):
        returns = returns.at[i].set(rews + returns[i+1]*gamma*masks[i+1])
    return returns

@jax.jit
def concat_trajectories(traj_list):
    return [jnp.concatenate(x, axis=0) for x in zip(*traj_list)]

@functools.partial(jax.jit)
def stack_experiences(exp_list):
    num_steps = exp_list[0].observations.shape[0]
    last_vals = exp_list[-1][3][-1]
    last_dones = exp_list[-1][4][-1]
    last_states = exp_list[-1][5][-1]
    observations = jnp.concatenate([x.observations for x in exp_list], axis=0)
    actions = jnp.concatenate([x.actions for x in exp_list], axis=0)
    rewards = jnp.concatenate([x.rewards for x in exp_list], axis=0)
    values = jnp.concatenate([x.values[:num_steps] for x in exp_list], axis=0)
    dones = jnp.concatenate([x.dones[:num_steps] for x in exp_list], axis=0)
    states = []
    for st in exp_list:
        states += st.states[:num_steps]
    values = jnp.append(values, last_vals[None], axis=0)
    dones = jnp.append(dones, last_dones[None], axis=0)
    states.append(last_states)
    return Experience(
        observations=observations,
        actions=actions,
        rewards=rewards,
        values=values,
        dones=dones,
        states=states
    )

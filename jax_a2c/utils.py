import functools
from typing import Any, Callable, Dict, Tuple
from collections import namedtuple
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
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
        observations=jnp.stack(observations_list),
        actions=jnp.stack(actions_list),
        rewards=jnp.stack(rewards_list),
        values=jnp.stack(values_list),
        dones=jnp.stack(dones_list),
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


def k_mc_rollouts_trajectories(prngkey, k_envs, experience, policy_fn, gamma, max_steps=1000):
    observations = experience.observations
    actions = experience.actions
    rewards = experience.rewards
    values = experience.values
    dones = experience.dones
    states = experience.states


    rep_returns = []
    rep_actions = []

    K = k_envs.num_envs // observations.shape[1]

    orig_mc_returns = get_mc_returns(rewards, dones, values[-1], gamma)
    # rint(orig_mc_returns.shape)

    iterations = 0
    
    for i, (obs, done, state) in enumerate(zip(observations, dones, states)):
        _, prngkey = jax.random.split(prngkey)
        next_ob = jnp.concatenate([obs for _ in range(K)])
        done = jnp.concatenate([done for _ in range(K)])
        state = state * K


        # COLLECTING ------------
        k_envs.set_state(state)
        observations_list = []
        actions_list = []
        rewards_list = []
        values_list = []
        ds = [done]

        cumdones = jnp.zeros(shape=(next_ob.shape[0],))

        for _ in range(max_steps):
            ob = next_ob
            _, prngkey = jax.random.split(prngkey)
            _, acts = policy_fn(prngkey, ob) 
            next_ob, rews, d, info = k_envs.step(acts)
            # observations_list.append(observations)
            actions_list.append(np.array(acts))
            rewards_list.append(rews)
            # values_list.append(values[..., 0])
            ds.append(d)
            iterations += 1
            cumdones += d
            if cumdones.all():
                break

        bootstrapped_values, _ = policy_fn(prngkey, ob) 
        values_list.append(bootstrapped_values[..., 0])
        ds = jnp.stack(ds)
        rewards_list = jnp.stack(rewards_list)

        kn_returns = process_rewards(ds, rewards_list, bootstrapped_values[..., 0], gamma, K)
        rep_returns.append(kn_returns)
        rep_actions.append(actions_list[0])

        # values = jnp.concatenate([values_list, values], axis=1)
        


    rep_observations = jnp.concatenate([observations for _ in range(K)], axis=1)
    rep_values = jnp.concatenate([values for _ in range(K)], axis=1)
    rep_actions = jnp.stack(rep_actions)
    rep_returns = jnp.stack(rep_returns)
    rep_advantages = rep_returns - rep_values[:-1]


    trajectories = (rep_observations, rep_actions, rep_returns, rep_advantages)
    trajectory_len = rep_observations.shape[0] * rep_observations.shape[1]
    trajectories = tuple(map(
        lambda x: np.reshape(x, (trajectory_len,) + x.shape[2:]), trajectories))

    old_trajectories = (observations, actions, orig_mc_returns, orig_mc_returns - values[:-1])
    old_trajectory_len = observations.shape[0] * observations.shape[1]
    old_trajectories = tuple(map(
        lambda x: np.reshape(x, (old_trajectory_len,) + x.shape[2:]), old_trajectories))

    trajectories = [jnp.concatenate([t1, t2], axis=0) for t1, t2 in zip(trajectories, old_trajectories)]

    return trajectories

@functools.partial(jax.jit, static_argnums=(3, 4))
def process_rewards(dones, rewards, bootstrapped_values, gamma, K):
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
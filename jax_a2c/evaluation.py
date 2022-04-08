from typing import Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np

import jax_a2c.env_utils
from jax_a2c.distributions import sample_action_from_normal as sample_action

def eval(
    apply_fn: Callable, 
    params: flax.core.frozen_dict, 
    env: jax_a2c.env_utils.SubprocVecEnv):

    apply_fn = jax.jit(apply_fn)
    env.training = False
    observation = env.reset()
    total_reward = []
    cumdones = jnp.zeros(shape=(observation.shape[0],))
    dones = [np.array(observation.shape[0]*[False])]
    for _ in range(1000):
        values, (action_means, action_log_stds) = apply_fn({'params': params}, observation)
        observation, reward, done, info = env.step(action_means)
        cumdones += done
        total_reward.append(env.old_reward) 
        dones.append(done)
        if cumdones.all():
            break
    masks = jnp.cumprod(1-jnp.array(dones), axis=0)[:-1]
    return observation, (jnp.array(total_reward)*masks).sum(axis=0).mean().item()


def q_eval(
    apply_fn: Callable, 
    params: flax.core.frozen_dict, 
    q_fn: Callable,
    q_params: flax.core.frozen_dict,
    env: jax_a2c.env_utils.SubprocVecEnv):

    N = 10

    apply_fn = jax.jit(apply_fn)
    q_fn = jax.jit(q_fn)

    env.training = False
    observation = env.reset()
    total_reward = []
    cumdones = jnp.zeros(shape=(observation.shape[0],))
    dones = [np.array(observation.shape[0]*[False])]

    prngkey = jax.random.PRNGKey(999)
    for _ in range(1000):
        prngkey, _ = jax.random.split(prngkey)
        # observation = jnp.concatenate([observation for _ in range(10)], axis=0)

        values, (action_means, action_log_stds) = apply_fn({'params': params}, observation)

        action_means_rep = jnp.concatenate([action_means for _ in range(N)], axis=0)
        action_log_stds_rep = jnp.concatenate([action_log_stds for _ in range(N)], axis=0)

        sampled_actions  = sample_action(prngkey, action_means_rep, action_log_stds_rep)
        sampled_actions = jnp.concatenate([sampled_actions, action_means], axis=0)

        rep_obs = jnp.concatenate([observation for _ in range(N+1)], axis=0)

        q_vals = q_fn({'params': q_params}, rep_obs, sampled_actions).reshape((N+1), rep_obs.shape[0]//(N+1))
        to_select = q_vals.argmax(0)
        sampled_actions = sampled_actions.reshape((N+1), sampled_actions.shape[0]//(N+1), sampled_actions.shape[-1])

        sampled_actions = jnp.take_along_axis(sampled_actions, indices=to_select[None, :, None], axis=0)[0]
        observation, reward, done, info = env.step(sampled_actions)
        cumdones += done
        total_reward.append(env.old_reward) 
        dones.append(done)
        if cumdones.all():
            break
    masks = jnp.cumprod(1-jnp.array(dones), axis=0)[:-1]
    return observation, (jnp.array(total_reward)*masks).sum(axis=0).mean().item()


def _q_policy_fn(prngkey, observations, params, apply_fn, q_params, q_fn):
    observations = jnp.concatenate([observations for _ in range(10)], axis=0)
    values, (means, log_stds) = apply_fn({'params': params}, observations)
    sampled_actions  = sample_action(prngkey, means, log_stds)# .reshape(10, observations.shape[0]//10, -1)
    q_vals = apply_fn({'params': params}, observations).reshape(10, observations.shape[0]//10, -1)
    to_select = q_vals.argmax(0)
    sampled_actions = jnp.take_along_axis(sampled_actions, indices=to_select[None, :, None], axis=0)[0]
    values = jnp.take_along_axis(values, indices=to_select[None, :, None], axis=0)[0]
    return values, sampled_actions
from typing import Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np

import jax_a2c.env_utils


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

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
    for _ in range(10000):
        values, logits = apply_fn({'params': params}, observation)
        observation, reward, done, info = env.step(logits.argmax(-1))
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
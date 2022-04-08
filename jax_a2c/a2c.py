import functools
from typing import Any, Callable, Dict, Optional, Tuple

import flax
import jax
import jax.numpy as jnp

from jax_a2c.distributions import evaluate_actions_norm as evaluate_actions
from jax_a2c.utils import PRNGKey

Array = Any

@functools.partial(jax.jit, static_argnums=(1,6,7,8,9,10,11))
def loss_fn(
    params: flax.core.frozen_dict, 
    apply_fn: Callable, 
    observations: Array, 
    actions: Array, 
    returns: Array, 
    prngkey: PRNGKey,
    q_fn: Callable,
    value_loss_coef: float = .5, 
    q_loss_coef: float = .5,
    entropy_coef: float = .01,
    normalize_advantages: bool = True, 
    q_updates: Optional[str] = None):
    action_logprobs, values, dist_entropy, log_stds, action_samples = evaluate_actions(
        params['policy_params'], 
        apply_fn, observations, actions, prngkey)
    advantages = returns - values
    if normalize_advantages:
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-6)
    q_loss = 0
    if q_updates == 'rep':
        q_loss = ((q_fn({'params': params['qf_params']}, observations, actions) - returns)**2).mean()
        q_loss += - q_loss_coef * q_fn(
                    jax.lax.stop_gradient({'params': params['qf_params']}), 
                    observations, 
                    action_samples).mean()
    elif q_updates == 'log':
        q_loss = ((q_fn({'params': params['qf_params']}, observations, actions) - returns)**2).mean()
        estimations = q_fn({'params': params['qf_params']}, observations, action_samples)
        estimated_advantages = estimations - values
        q_loss += - (jax.lax.stop_gradient(estimated_advantages) * action_logprobs).mean()
    elif q_updates == 'just_q':
        q_loss = ((q_fn({'params': params['qf_params']}, observations, actions) - returns)**2).mean()

    policy_loss = - (jax.lax.stop_gradient(advantages) * action_logprobs).mean()
    value_loss = ((returns - values)**2).mean()
    loss = value_loss_coef*value_loss + policy_loss - entropy_coef*dist_entropy + q_loss
    return loss, dict(
        value_loss=value_loss, 
        policy_loss=policy_loss, 
        dist_entropy=dist_entropy, 
        advantages_max = jnp.abs(advantages).max(),
        min_std=jnp.exp(log_stds).min(),
        q_loss=q_loss)

@functools.partial(jax.jit, static_argnums=(3,4,5,6,7))
def step(state, trajectories, prngkey,
value_loss_coef=.5, 
q_loss_coef=.5,
entropy_coef=.01, normalize_advantages=True, q_updates: str = None):
    observations, actions, returns, advantages = trajectories
    (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, 
        state.apply_fn, 
        observations, 
        actions, 
        returns,
        prngkey,
        state.q_fn,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        normalize_advantages=normalize_advantages, 
        q_updates=q_updates,
        q_loss_coef=q_loss_coef)
    new_state = state.apply_gradients(grads=grads)
    return new_state, (loss, loss_dict)
    
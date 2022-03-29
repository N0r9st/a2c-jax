import functools
from collections import namedtuple
from typing import Any, Callable, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

import jax_a2c.env_utils
from jax_a2c.distributions import evaluate_actions_norm as evaluate_actions

Array = Any
PRNGKey = Any
ModelClass = Any

Array = Any

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


@functools.partial(jax.jit, static_argnums=(1,5,6,7))
def loss_fn(
    params: flax.core.frozen_dict, 
    apply_fn: Callable, 
    observations: Array, 
    actions: Array, 
    returns: Array, 
    value_loss_coef: float = .5, 
    entropy_coef: float = .01,
    normalize_advantages: bool = True):
    action_logprobs, values, dist_entropy, log_stds = evaluate_actions(params, apply_fn, observations, actions)
    advantages = returns - values
    if normalize_advantages:
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-6)
    policy_loss = - (jax.lax.stop_gradient(advantages) * action_logprobs).mean()
    value_loss = ((returns - values)**2).mean()
    loss = value_loss_coef*value_loss + policy_loss - entropy_coef*dist_entropy
    return loss, dict(
        value_loss=value_loss, 
        policy_loss=policy_loss, 
        dist_entropy=dist_entropy, 
        advantages_max = jnp.abs(advantages).max(),
        min_std=jnp.exp(log_stds).min())

@functools.partial(jax.jit, static_argnums=(2,3,4))
def step(state, trajectories, value_loss_coef=.5, entropy_coef=.01, normalize_advantages=True):
    observations, actions, returns, advantages = trajectories
    (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, 
        state.apply_fn, 
        observations, 
        actions, 
        returns,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        normalize_advantages=normalize_advantages)
    new_state = state.apply_gradients(grads=grads)
    return new_state, (loss, loss_dict)
    
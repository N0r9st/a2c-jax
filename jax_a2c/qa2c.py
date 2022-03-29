import functools
from collections import namedtuple
from typing import Any, Callable, Dict, Tuple

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from flax import struct, core

import jax_a2c.env_utils

Array = Any
PRNGKey = Any
ModelClass = Any


class QTrainState(TrainState):
    q_fn: Callable  = struct.field(pytree_node=False)
    

def create_train_state(
    prngkey: PRNGKey, 
    model: ModelClass,
    q_model: ModelClass,
    envs: jax_a2c.env_utils.SubprocVecEnv,
    learning_rate: float, 
    decaying_lr: bool, 
    max_norm: float,
    decay: float,
    eps: float,
    train_steps: int = 0) -> TrainState:

    dummy_input = envs.reset()
    dummy_action = np.stack([envs.action_space.sample() for _ in range(envs.num_envs)])

    variables = model.init(prngkey, dummy_input)
    params = variables['params']

    prngkey, _ = jax.random.split(prngkey)
    q_variables = q_model.init(prngkey, dummy_input, dummy_action)
    q_params = q_variables['params']

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
        apply_fn=model.apply,
        params={'params': params, 'q_params': q_params},
        q_fn=q_model.apply,
        tx=tx)

    return state


@functools.partial(jax.jit, static_argnums=(1,2,7,8,9,10,11))
def loss_fn(
    params: flax.core.frozen_dict, 
    apply_fn: Callable, 
    q_fn: Callable,
    observations: Array, 
    actions: Array, 
    returns: Array, 
    prngkey: PRNGKey,
    value_loss_coef: float = .5, 
    q_value_loss_coef: float = .5,
    entropy_coef: float = .01,
    normalize_advantages: bool = True,
    q_value_target: bool = False):
    action_logprobs, values, q_values, dist_entropy, log_stds = evaluate_actions(params, apply_fn, q_fn, observations, actions)
    
    if normalize_advantages:
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-6)
    if q_value_target:
        # advantages = q_values - values
        values = evaluate_v(prngkey, params, apply_fn, q_fn, observations, num=100)

    advantages = returns - values

    value_loss = ((returns - values)**2).mean()
    q_value_loss = ((returns - q_values)**2).mean()
    policy_loss = - (jax.lax.stop_gradient(advantages) * action_logprobs).mean()
    loss = value_loss_coef*value_loss + q_value_loss_coef*q_value_loss + policy_loss - entropy_coef*dist_entropy
    return loss, dict(
        value_loss=value_loss, 
        policy_loss=policy_loss, 
        dist_entropy=dist_entropy, 
        advantages_max = jnp.abs(advantages).max(),
        min_std=jnp.exp(log_stds).min())

@functools.partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def step(
    state, 
    trajectories, 
    prngkey,
    value_loss_coef=.5, 
    q_value_loss_coef=.5, 
    entropy_coef=.01, 
    normalize_advantages=True,
    q_value_target=False):
    observations, actions, returns, advantages = trajectories
    (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, 
        state.apply_fn, 
        state.q_fn,
        observations, 
        actions, 
        returns,
        prngkey,
        value_loss_coef=value_loss_coef,
        q_value_loss_coef=q_value_loss_coef,
        entropy_coef=entropy_coef,
        normalize_advantages=normalize_advantages,
        q_value_target=q_value_target)
    new_state = state.apply_gradients(grads=grads)
    return new_state, (loss, loss_dict)
    

def evaluate_actions(params, apply_fn, q_fn, observations, actions):
    values, (means, log_stds) = apply_fn({'params': params['params']}, observations)
    q_values = q_fn({'params': params['q_params']}, observations, actions)
    stds = jnp.exp(log_stds)
    pre_tanh_logprobs = -(actions-means)**2/(2*stds**2) - jnp.log(2*jnp.pi)/2 - log_stds
    action_logprobs = (pre_tanh_logprobs).sum(axis=-1)
    dist_entropy = (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_stds).mean()
    return action_logprobs, values[..., 0], q_values[..., 0], dist_entropy, log_stds  

    


def evaluate_v(prngkey, params, apply_fn, q_fn, observations, num=100):
    values, (means, log_stds) = apply_fn({'params': params['params']}, observations)
    batch_size, action_shape = means.shape

    observations = jnp.stack([observations for _ in range(num)])
    means = jnp.stack([means for _ in range(num)])
    log_stds = jnp.stack([log_stds for _ in range(num)])

    actions = means + jnp.exp(log_stds) * jax.random.normal(prngkey, shape=means.shape)
    actions = actions.reshape(num*batch_size, action_shape)
    q_s = q_fn({'params': params['q_params']}, observations.reshape(num*batch_size, -1), actions)
    v_s = q_s.reshape(num, batch_size, -1).mean(0)
    assert v_s.shape == values.shape, f"{v_s.shape} != {values.shape}"
    return v_s
import functools
from typing import Any, Callable, Dict, Optional, Tuple

import flax
import jax
import jax.numpy as jnp

from jax_a2c.distributions import evaluate_actions_norm as evaluate_actions
from jax_a2c.distributions import sample_action_from_normal as sample_action
from jax_a2c.utils import (PRNGKey, calculate_action_logprobs, process_mc_rollouts,
                           vmap_process_mc_rollouts,
                           vmap_process_rewards_with_entropy, process_experience_with_entropy)

Array = Any

def p_loss_fn(
    params: flax.core.frozen_dict, 
    apply_fn: Callable,  
    oar,
    prngkey: PRNGKey,
    q_fn: Callable,
    constant_params,
    ):

    observations = oar['observations']
    actions = oar['actions']
    returns = oar['returns']

    action_logprobs, sampled_action_logprobs, values, dist_entropy, log_stds, action_samples = evaluate_actions(
        params['policy_params'], 
        apply_fn, observations, actions, prngkey)
    #--------------------------------
    #          ADVANTAGES
    #--------------------------------
    advantages = jax.lax.stop_gradient(returns - values)

    #----------------------------------------
    #               BASIC UPDATE
    #----------------------------------------
    loss_dict = {}

    policy_loss = - (advantages * action_logprobs).mean()
    value_loss = ((returns - values)**2).mean()
    qp_loss = jnp.array(0)
    if constant_params['q_updates'] == 'rep':
        qp_loss = - (q_fn(jax.lax.stop_gradient({'params': params['qf_params']}), observations, action_samples).mean() + \
            constant_params['alpha']*dist_entropy)

    elif constant_params['q_updates'] == 'log':
        if constant_params['use_samples_for_log_update']:
            sampled_estimations = q_fn({'params': params['qf_params']}, observations, action_samples)
            q_logprobs = sampled_action_logprobs
        else:
            sampled_estimations = q_fn({'params': params['qf_params']}, observations, actions)
            q_logprobs = action_logprobs
        estimated_advantages = sampled_estimations - values
        qp_loss = - (jax.lax.stop_gradient(estimated_advantages) * q_logprobs).mean()


    loss = constant_params['value_loss_coef']*value_loss + policy_loss - constant_params['entropy_coef']*dist_entropy + \
        constant_params['q_loss_coef'] * qp_loss

    loss_dict.update(
        value_loss=value_loss, 
        policy_loss=policy_loss, 
        dist_entropy=dist_entropy, 
        advantages_max = jnp.abs(advantages).max(),
        min_std=jnp.exp(log_stds).min(),
        mean_returns=returns.mean(),
        std_returns=returns.std(),
        mean_logprog=action_logprobs.mean(),
        std_logprog=action_logprobs.std(),
        )
    return loss, loss_dict

@functools.partial(jax.jit, static_argnums=(3,))
def p_step(state, data_tuple, prngkey,
    constant_params):
    
    (loss, loss_dict), grads = jax.value_and_grad(p_loss_fn, has_aux=True)(
        state.params, 
        state.apply_fn, 
        data_tuple,
        prngkey,
        state.q_fn,
        constant_params,)
    new_state = state.apply_gradients(grads=grads)
    return new_state, (loss, loss_dict)
    
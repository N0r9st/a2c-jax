import functools
from typing import Any, Callable, Dict, Optional, Tuple

import flax
import jax
import jax.numpy as jnp

from jax_a2c.distributions import evaluate_actions_norm as evaluate_actions, sample_acts_for_obs
from jax_a2c.distributions import sample_action_from_normal as sample_action
from jax_a2c.utils import (PRNGKey, calculate_action_logprobs, process_mc_rollouts,
                           vmap_process_mc_rollouts,
                           vmap_process_rewards_with_entropy, process_experience_with_entropy)

Array = Any

def p_loss_fn(
    params: flax.core.frozen_dict, 
    apply_fn: Callable,  
    data_dict,
    prngkey: PRNGKey,
    q_fn: Callable,
    constant_params,
    ):
    oar = data_dict['oar']
    observations = oar['observations']
    actions = oar['actions']
    returns = oar['returns']

    action_logprobs, sampled_action_logprobs, values, dist_entropy, log_stds, action_samples = evaluate_actions(
        params['policy_params'], 
        apply_fn, observations, actions, prngkey)

    if constant_params['return_in_remaining']:
        q_observations = data_dict['not_sampled_observations']
        q_action_samples, q_logprobs, q_values = sample_acts_for_obs(
            params['policy_params'], apply_fn, 
            prngkey, q_observations, constant_params['K'], constant_params['logstd_stopgrad'])
        q_observations = jnp.concatenate([q_observations]*constant_params['K'], axis=0)
    else:
        q_observations = observations
        q_action_samples, q_logprobs, q_values = action_samples, sampled_action_logprobs, values
    #--------------------------------
    #          ADVANTAGES
    #--------------------------------
    advantages = jax.lax.stop_gradient(returns - values)

    #----------------------------------------
    #               BASIC UPDATE
    #----------------------------------------
    loss_dict = {}
    if constant_params['equal_importance_ploss']:
        policy_loss = - (advantages * action_logprobs)
    else:
        policy_loss = - (advantages * action_logprobs).mean()

    value_loss = ((returns - values)**2).mean()
    qp_loss = jnp.array(0)
    if constant_params['q_updates'] == 'rep':

        qp_loss = - (q_fn(
            jax.lax.stop_gradient({'params': params['qf_params']}), q_observations, q_action_samples
            ).mean() +\
                 constant_params['alpha']*dist_entropy)

    elif constant_params['q_updates'] == 'log':
        
        q_sampled_estimations = q_fn({'params': params['qf_params']}, q_observations, q_action_samples)
        q_estimated_advantages = q_sampled_estimations - q_values

        if constant_params['equal_importance_ploss']:
            qp_loss = - constant_params['q_loss_coef'] * (jax.lax.stop_gradient(q_estimated_advantages) * q_logprobs)
            policy_loss = jnp.concatenate([policy_loss, qp_loss])
        else:
            policy_loss += - constant_params['q_loss_coef'] * \
                (jax.lax.stop_gradient(q_estimated_advantages) * q_logprobs).mean()
    
    if constant_params['equal_importance_ploss']:
        policy_loss = policy_loss.mean()
    loss = constant_params['value_loss_coef']*value_loss + policy_loss - constant_params['entropy_coef'] * dist_entropy

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
def p_step(state, data_dict, prngkey,
    constant_params):
    
    (loss, loss_dict), grads = jax.value_and_grad(p_loss_fn, has_aux=True)(
        state.params, 
        state.apply_fn, 
        data_dict,
        prngkey,
        state.q_fn,
        constant_params,)
    new_state = state.apply_gradients(grads=grads)
    return new_state, (loss, loss_dict)
    
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

def q_loss_fn(
    params: flax.core.frozen_dict, 
    apply_fn: Callable,  
    oar: dict,
    prngkey: PRNGKey,
    q_fn: Callable,
    constant_params,
    ):


    observations = oar['observations']
    actions = oar['actions']
    returns = oar['returns']
    
    q_estimations = q_fn({'params': params['qf_params']}, observations, actions)
    q_loss = ((q_estimations - jax.lax.stop_gradient(returns))**2).mean()

    rand_actions = jax.random.uniform(prngkey, shape=(len(observations), actions.shape[-1]))
    rand_q_estimations = q_fn({'params': params['qf_params']}, observations, rand_actions)

    loss = constant_params['q_loss_multiplier'] * q_loss

    loss_dict = {}
    loss_dict.update(
        q_loss=q_loss,
        current_q=q_estimations.mean(),
        baseline_q=rand_q_estimations.mean(),
        difference_q =q_estimations.mean() - rand_q_estimations.mean(),
        )
    return loss, loss_dict

@functools.partial(jax.jit, static_argnums=(3,))
def q_step(state, data_tuple, prngkey,
    constant_params):
    
    (loss, loss_dict), grads = jax.value_and_grad(q_loss_fn, has_aux=True)(
        state.params, 
        state.apply_fn, 
        data_tuple,
        prngkey,
        state.q_fn,
        constant_params,)
    new_state = state.apply_gradients(grads=grads)
    return new_state, (loss, loss_dict)


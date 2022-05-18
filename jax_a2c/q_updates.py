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
    q_loss = ((q_estimations - returns)**2).mean()


    loss = constant_params['q_loss_multiplier'] * q_loss

    loss_dict = {}
    loss_dict.update(
        q_loss=q_loss,
        current_q=q_estimations.mean(),
        # baseline_q=rand_q_estimations.mean(),
        # difference_q =q_estimations.mean() - rand_q_estimations.mean(),
        )
    return loss, loss_dict

@functools.partial(jax.jit, static_argnums=(3,))
def q_step(state, train_oar, prngkey,
    constant_params):
    loss = jnp.array(0)
    loss_dict = {}
    for _ in range(constant_params['qf_update_epochs']):
        prngkey, _ = jax.random.split(prngkey)
        batches = get_batches(train_oar, constant_params['qf_update_batch_size'], prngkey)
        for batch in batches:
            (loss, loss_dict), grads = jax.value_and_grad(q_loss_fn, has_aux=True)(
                state.params, 
                state.apply_fn, 
                batch,
                prngkey,
                state.q_fn,
                constant_params,)
            state = state.apply_gradients(grads=grads)

    # loss_dict.update(test_qf(prngkey, train_oar, test_oar, state.q_fn, state.params))

    return state, (loss, loss_dict)

def get_batches(oar, batch_size, prngkey):
    if batch_size <0:
        return [oar]
    oar = {k: jax.random.permutation(prngkey, v, independent=True) for k, v in oar.items()}
    num_obs = len(oar['observations'])
    batches = []
    for i in range(0, num_obs, batch_size):
        batches.append({k: v[i:i+batch_size] for k, v in oar.items()})
    return batches

def test_qf(prngkey, train_oar, test_oar, q_fn, params):
    train_observations = train_oar['observations']
    train_actions = train_oar['actions']
    train_returns = train_oar['returns']

    test_observations = test_oar['observations']
    test_actions = test_oar['actions']
    test_returns = test_oar['returns']
    
    q_train_estimations = q_fn({'params': params['qf_params']}, train_observations, train_actions)
    q_train_loss = ((q_train_estimations - train_returns)**2).mean()

    q_test_estimations = q_fn({'params': params['qf_params']}, test_observations, test_actions)
    q_test_loss = ((q_test_estimations - test_returns)**2).mean()


    train_rand_actions = jax.random.uniform(prngkey, shape=(len(train_observations), train_actions.shape[-1]))
    train_rand_q_estimations = q_fn({'params': params['qf_params']}, train_observations, train_rand_actions)
    train_q_diff = (q_train_estimations - train_rand_q_estimations).mean()
    

    test_rand_actions = jax.random.uniform(prngkey, shape=(len(test_observations), test_actions.shape[-1]))
    test_rand_q_estimations = q_fn({'params': params['qf_params']}, test_observations, test_rand_actions)
    test_q_diff = (q_test_estimations - test_rand_q_estimations).mean()

    return dict(
        q_train_loss=q_train_loss,
        q_test_loss=q_test_loss,
        q_train_test_loss_diff = q_test_loss - q_train_loss,
        train_q_diff=train_q_diff,
        test_q_diff=test_q_diff,

    )

def train_test_split(oar, prngkey, test_ratio, num_train_samples):
    num_test = int(num_train_samples*test_ratio)
    test_choices = jax.random.choice(prngkey, num_train_samples, shape=(num_test,), replace=False)
    test_mask = jnp.zeros((num_train_samples,), dtype=bool).at[test_choices].set(True)
    # oar = {k: jax.random.shuffle(prngkey, v) for k, v in oar.items()}
    return (
        {k: v[jnp.logical_not(test_mask)] for k, v in oar.items()},
        {k: v[test_mask] for k, v in oar.items()}, 
        )
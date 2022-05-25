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

# @functools.partial(jax.jit, static_argnums=(4,))
def q_step(state, train_oar, test_oar, prngkey,
    constant_params, jit_q_fn, ):
    loss = jnp.array(0)
    loss_dict = {}

    
    if not constant_params['use_q_tolerance']:
        for _ in range(constant_params['qf_update_epochs']):
            prngkey, _ = jax.random.split(prngkey)
            batches = get_batches(
                train_oar, constant_params['qf_update_batch_size'], 
                prngkey, constant_params['q_train_len'])
            for batch in batches:
                state, _ = q_microstep(state, batch, prngkey, constant_params)
            q_loss_dict = test_qf(prngkey, train_oar, test_oar, jit_q_fn, state.params)
    else:
        min_test_loss = float('inf')
        tolerance = 0
        epoch_count = 0
        for _ in range(512):
            prngkey, _ = jax.random.split(prngkey)
            batches = get_batches(
                train_oar, constant_params['qf_update_batch_size'], 
                prngkey, constant_params['q_train_len'])
            for batch in batches:
                state, _ = q_microstep(state, batch, prngkey, constant_params)
            epoch_count += 1

            q_loss_dict = test_qf(prngkey, train_oar, test_oar, jit_q_fn, state.params)
            if q_loss_dict['q_test_loss'] < min_test_loss:
                best_state = state
                min_test_loss = q_loss_dict['q_test_loss'] 
                tolerance = 0
            else:
                tolerance += 1
            
            if tolerance > constant_params['max_q_tolerance']:
                epoch_count = epoch_count - tolerance
                q_loss_dict['epoch_count'] = jnp.array(epoch_count)
                state = best_state
                break

        if constant_params['full_data_for_q_update']:
            oar = merge_train_test(train_oar, test_oar)
            for _ in range(epoch_count):
                prngkey, _ = jax.random.split(prngkey)
                batches = get_batches(
                    oar, constant_params['qf_update_batch_size'], 
                    prngkey, constant_params['q_train_len'])
                for batch in batches:
                    state, _ = q_microstep(state, batch, prngkey, constant_params)
                q_loss_dict = test_qf(prngkey, train_oar, test_oar, jit_q_fn, state.params)

    return state, (loss, q_loss_dict)

@jax.jit
def merge_train_test(train_oar, test_oar):
    oar = {}
    for k in train_oar:
        oar[k] = jnp.concatenate((train_oar[k], test_oar[k]), axis=0)
    return oar

@functools.partial(jax.jit, static_argnums=(3,4,5,6,))
def q_microstep(state, batch, prngkey,
    constant_params):
    (loss, loss_dict), grads = jax.value_and_grad(q_loss_fn, has_aux=True)(
        state.params, 
        state.apply_fn, 
        batch,
        prngkey,
        state.q_fn,
        constant_params,)
    new_state = state.apply_gradients(grads=grads)
    return new_state, (loss, loss_dict)
    
@functools.partial(jax.jit,static_argnums=(1,3,))
def get_batches(oar, batch_size, prngkey, data_len):
    if batch_size <0:
        return [oar]
    perm = jax.random.permutation(prngkey, data_len)
    oar = {k: v[perm] for k, v in oar.items()}
    batches = []
    for i in range(0, data_len, batch_size):
        batches.append({k: v[i:i+batch_size] for k, v in oar.items()})
    return batches

@functools.partial(jax.jit,static_argnums=(3,))
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


    train_rand_actions = jax.random.uniform(prngkey, shape=train_actions.shape)
    train_rand_q_estimations = q_fn({'params': params['qf_params']}, train_observations, train_rand_actions)
    train_q_diff = (q_train_estimations - train_rand_q_estimations).mean()
    

    test_rand_actions = jax.random.uniform(prngkey, shape=test_actions.shape)
    test_rand_q_estimations = q_fn({'params': params['qf_params']}, test_observations, test_rand_actions)
    test_q_diff = (q_test_estimations - test_rand_q_estimations).mean()

    return dict(
        q_train_loss=q_train_loss,
        q_test_loss=q_test_loss,
        q_train_test_loss_diff = q_test_loss - q_train_loss,
        train_q_diff=train_q_diff,
        test_q_diff=test_q_diff,

    )

# @functools.partial(jax.jit,static_argnums=(2,3))
def train_test_split(oar, prngkey, test_ratio, num_train_samples):
    num_test = int(num_train_samples*test_ratio)
    test_choices = jax.random.choice(prngkey, num_train_samples, shape=(num_test,), replace=False)
    test_mask = jnp.zeros((num_train_samples,), dtype=bool).at[test_choices].set(True)
    # oar = {k: jax.random.shuffle(prngkey, v) for k, v in oar.items()}
    return (
        {k: v[jnp.logical_not(test_mask)] for k, v in oar.items()},
        {k: v[test_mask] for k, v in oar.items()}, 
        )

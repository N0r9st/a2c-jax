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
        )
    return loss, loss_dict

# @functools.partial(jax.jit, static_argnums=(4,))
def q_step(state, train_oar, test_oar, prngkey,
    constant_params, jit_q_fn, ):
    loss = jnp.array(0)
        
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
                best_q_loss_dict = q_loss_dict
                best_state = state
                min_test_loss = q_loss_dict['q_test_loss'] 
                tolerance = 0
            else:
                tolerance += 1
            
            if tolerance > constant_params['max_q_tolerance']:
                epoch_count = epoch_count - tolerance
                state = best_state
                q_loss_dict = best_q_loss_dict
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
                # q_loss_dict = test_qf(prngkey, train_oar, test_oar, jit_q_fn, state.params)

        q_loss_dict['epoch_count'] = jnp.array(epoch_count)

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
def train_test_split(oar, prngkey, test_ratio, num_train_samples, num_test=None):
    if num_test is None:
        num_test = int(num_train_samples*test_ratio)
    test_choices = jax.random.choice(prngkey, num_train_samples, shape=(num_test,), replace=False)
    test_mask = jnp.zeros((num_train_samples,), dtype=bool).at[test_choices].set(True)
    # oar = {k: jax.random.shuffle(prngkey, v) for k, v in oar.items()}
    return (
        {k: v[jnp.logical_not(test_mask)] for k, v in oar.items()},
        {k: v[test_mask] for k, v in oar.items()}, 
        )

def train_test_split_k_repeat(oar, prngkey, test_ratio, num_train_samples, k, nw, test_choices=None):
    """ Unlike train_test_split, splits oar data 
    """
    oar = groub_by_repeats(oar, k, nw)
    num_diff_states = oar['observations'].shape[1]

    num_test = int(num_diff_states*test_ratio)

    if test_choices is None:
        test_choices = jax.random.choice(prngkey, num_diff_states, shape=(num_test,), replace=False)
    test_mask = jnp.zeros((num_diff_states,), dtype=bool).at[test_choices].set(True)
    # oar = {k: jax.random.shuffle(prngkey, v) for k, v in oar.items()}
    oar_train = {k: v[:, jnp.logical_not(test_mask)] for k, v in oar.items()}
    oar_test = {k: v[:, test_mask] for k, v in oar.items()}
    oar_train = jax.tree_util.tree_map(lambda x: x.reshape((x.shape[0]*x.shape[1],) + x.shape[2:]), oar_train)
    oar_test = jax.tree_util.tree_map(lambda x: x.reshape((x.shape[0]*x.shape[1],) + x.shape[2:]), oar_test)
    return oar_train, oar_test, test_choices

@functools.partial(jax.jit, static_argnames=('k', 'nw'))
def groub_by_repeats(oar, k, nw):
    # oar = jax.tree_util.tree_map(lambda x: x.reshape((nw, k, x.shape[0]//nw//k) + x.shape[1:]), oar)
    # oar = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), oar)
    # oar = jax.tree_util.tree_map(lambda x: x.reshape((k, x.shape[2]*nw,) + x.shape[3:]), oar)
    oar = jax.tree_util.tree_map(lambda x: group_by_repeats_single(x, k, nw), oar)
    return oar

@functools.partial(jax.jit, static_argnames=('k', 'nw'))
def group_by_repeats_single(x, k, nw):
    x = x.reshape((nw, k, x.shape[0]//nw//k) + x.shape[1:])
    x = jnp.swapaxes(x, 0, 1)
    x = x.reshape((k, x.shape[2]*nw,) + x.shape[3:])
    return x

@jax.vmap
def masks_along_0(arr, ind):
    return arr[ind]

def apply_no_sampling_masks_single(arr, no_sampling_masks, nw, num_steps, num_envs):
    arr = arr.reshape((nw, num_steps//nw * num_envs,) + arr.shape[1:])
    arr = masks_along_0(arr, no_sampling_masks)
    arr = arr.reshape((arr.shape[0]*arr.shape[1],) + arr.shape[2:])
    return arr

@functools.partial(jax.jit, static_argnames=('nw', 'num_steps', 'num_envs'))
def apply_no_sampling_masks(oar, no_sampling_masks, nw, num_steps, num_envs):
    oar = jax.tree_util.tree_map(
        lambda x: apply_no_sampling_masks_single(x, no_sampling_masks, nw, num_steps, num_envs), 
        oar)
    return oar


def general_train_test_split(
    base_oar, 
    mc_oar, 
    negative_oar, 
    sampling_masks,
    no_sampling_masks,
    prngkey, 
    test_ratio, 
    k, nw, num_steps, num_envs, use_base_traj_for_q, full_tt_split, new_full_tt_split):
    """
    flags: use_base_traj_for_q, full_tt_split
    """
    if mc_oar is None:
        q_train_oar, q_test_oar = train_test_split(
                base_oar,
                prngkey, 
                test_ratio, len(base_oar['observations']),)
        return q_train_oar, q_test_oar
        
    if full_tt_split:
        q_train_oar, q_test_oar, test_choices = train_test_split_k_repeat(
            mc_oar,
            prngkey, 
            test_ratio, len(mc_oar['observations']), k=k, nw=nw)
        if negative_oar is not None:
            q_neg_train_oar, q_neg_test_oar, _ = train_test_split_k_repeat(
                negative_oar,
                prngkey, 
                test_ratio, len(negative_oar['observations']), k=k, nw=nw, test_choices=test_choices)
            q_train_oar = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), q_train_oar, q_neg_train_oar)
            q_test_oar = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), q_test_oar, q_neg_test_oar)
        if use_base_traj_for_q:
            q_base_train_oar, q_base_test_oar = train_test_split(
                base_oar,
                prngkey, 
                test_ratio, len(base_oar['observations']),)
            q_train_oar = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), q_train_oar, q_base_train_oar)
            q_test_oar = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), q_test_oar, q_base_test_oar)
    elif new_full_tt_split:
        masked_base_oar = apply_no_sampling_masks(base_oar, no_sampling_masks, nw, num_steps, num_envs)
        takenfork_base_oar = apply_no_sampling_masks(base_oar, sampling_masks, nw, num_steps, num_envs)
        not_taken_base_oar, q_test_oar = train_test_split(
            masked_base_oar,
            prngkey, 
            test_ratio, len(masked_base_oar['observations']), num_test=int(test_ratio*len(masked_base_oar['observations'])))
        not_taken_base_oar = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), not_taken_base_oar, takenfork_base_oar)
        q_train_oar = mc_oar
        if negative_oar is not None:
            q_train_oar = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), q_train_oar, negative_oar)
        if use_base_traj_for_q:
            q_train_oar = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), q_train_oar, not_taken_base_oar)
    else:
        q_train_oar, q_test_oar = train_test_split(
            mc_oar,
            prngkey, 
            test_ratio, len(mc_oar['observations']),)
        if negative_oar is not None:
            q_neg_train_oar, q_neg_test_oar = train_test_split(
                negative_oar,
                prngkey, 
                test_ratio, len(negative_oar['observations']),)
            q_train_oar = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), q_train_oar, q_neg_train_oar)
            q_test_oar = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), q_test_oar, q_neg_test_oar)
        if use_base_traj_for_q:
            q_base_train_oar, q_base_test_oar = train_test_split(
                base_oar,
                prngkey, 
                test_ratio, len(base_oar['observations']),)
            q_train_oar = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), q_train_oar, q_base_train_oar)
            q_test_oar = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x, axis=0), q_test_oar, q_base_test_oar)

    return q_train_oar, q_test_oar


@jax.vmap
def apply_mask(obs_pack, mask):
    return obs_pack[mask]

@jax.jit
@jax.vmap
def apply_mask_dict(oar_pack, mask):
    return jax.tree_util.tree_map(lambda x: x[mask], oar_pack)

# @functools.partial(jax.jit, static_argnames=('nw',))
def train_test_base_separateacts(base_oar, sampling_masks, no_sampling_masks, test_choices, non_test_choices, nw):
    """ separates base_oar on train and test so no test instances are in test_choices
    """
    base_oar = jax.tree_util.tree_map(lambda x: x.reshape((nw, x.shape[0]//nw,) + x.shape[1:]), base_oar)
    can_use_oar_nonsampled = apply_mask_dict(base_oar, no_sampling_masks)
    can_oar_sampled = apply_mask_dict(base_oar, sampling_masks)

    # can_use_oar_sampled = apply_mask_dict(can_oar_sampled, non_test_choices)
    # cant_use_oar = apply_mask_dict(can_oar_sampled, test_choices)
    can_use_oar_sampled = jax.tree_util.tree_map(lambda x: x[:, non_test_choices], can_oar_sampled)
    cant_use_oar = jax.tree_util.tree_map(lambda x: x[:, test_choices], can_oar_sampled)
    
    can_use_oar_nonsampled = jax.tree_util.tree_map(lambda x: x.reshape(-1, x.shape[-1]), can_use_oar_nonsampled)
    can_use_oar_sampled = jax.tree_util.tree_map(lambda x: x.reshape(-1, x.shape[-1]), can_use_oar_sampled)
    cant_use_oar = jax.tree_util.tree_map(lambda x: x.reshape(-1, x.shape[-1]), cant_use_oar)
    return can_use_oar_nonsampled, can_use_oar_sampled, cant_use_oar
    
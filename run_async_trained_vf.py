import functools
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import itertools
import multiprocessing as mp
import time
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

import wandb
from jax_a2c.a2c import p_step
from jax_a2c.distributions import sample_action_from_normal as sample_action
from jax_a2c.env_utils import DummySubprocVecEnv, make_vec_env, run_workers
from jax_a2c.evaluation import eval, q_eval
from jax_a2c.km_mc_traj import km_mc_rollouts
from jax_a2c.policy import (DGPolicy, DiagGaussianPolicy,
                            DiagGaussianStateDependentPolicy, QFunction,
                            QS0Function, QS1Function, VFunction)
from jax_a2c.q_updates import (general_train_test_split, q_step, test_qf,
                               train_test_split, train_test_split_k_repeat)
from jax_a2c.saving import load_state, save_state
from jax_a2c.utils import (Experience, calculate_interactions_per_epoch,
                           collect_experience, concat_trajectories, collect_experience_manyenvs,
                           create_train_state, process_base_rollout_output,
                           process_experience, process_mc_rollout_output,
                           process_rollout_output, select_random_states,
                           stack_experiences)

POLICY_CLASSES = {
    'DiagGaussianPolicy': DiagGaussianPolicy, 
    'DiagGaussianStateDependentPolicy': DiagGaussianStateDependentPolicy,
    "DGPolicy": DGPolicy,
}

QF_CLASSES = {
    'QFunction': QFunction, 
    'QS0Function': QS0Function,
    "QS1Function": QS1Function,
}
def _policy_fn(prngkey, observation, params, apply_fn, determenistic=False):
    means, log_stds = apply_fn({'params': params}, observation)
    sampled_actions  = means if determenistic else sample_action(prngkey, means, log_stds)
    return sampled_actions

def _value_and_policy_fn(prngkey, observation, params, vf_params, apply_fn, v_fn, determenistic=False):
    means, log_stds = apply_fn({'params': params}, observation)
    values = v_fn({'params': vf_params}, observation)
    sampled_actions  = means if determenistic else sample_action(prngkey, means, log_stds)
    return values, sampled_actions

def _worker(remote, k_remotes, parent_remote, spaces, device, add_args) -> None:
    print('D:', device)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    parent_remote.close()
    k_envs = DummySubprocVecEnv(remotes=k_remotes)
    
    k_envs.observation_space, k_envs.action_space = spaces
    k_envs = VecNormalize(k_envs, training=False)
    km_mc_rollouts_ = functools.partial(km_mc_rollouts, k_envs=k_envs)

    _policy_fn = jax.jit(add_args['policy_fn'], static_argnames=('determenistic',))
    v_fn = jax.jit(add_args['v_fn'])

    while True:
        try:
            args = remote.recv()
            k_envs.obs_rms = args.pop('train_obs_rms')
            k_envs.ret_rms = args.pop('train_ret_rms')
            policy_fn = functools.partial(_policy_fn, **(args.pop('policy_fn')))
            out = km_mc_rollouts_(policy_fn=policy_fn, v_fn=v_fn, vf_params=args.pop('vf_params'), **args)
            remote.send(out)
        except EOFError:
            break

def main(args: dict):
    args['async'] = True
    if not args['split_between_devices']:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['device']

    num_transition_steps = args['num_timesteps']//(args['num_envs'] * args['num_steps'])
    wandb_run_id = None
    start_update = 0
    timestep = 0
    epoch_times = []
    epoch_time = 0
    eval_return = 0
    q_eval_return = 0

    envs = make_vec_env(
        name=args['env_name'], 
        num=args['num_envs'], 
        norm_r=args['norm_r'], 
        norm_obs=args['norm_obs'],
        ctx=ctx,
        wrapper_params=args['wrappers'],)
        
    k_envs_fn = functools.partial(make_vec_env,
        name=args['env_name'], 
        num=args['num_k_envs'], 
        norm_r=args['norm_r'], 
        norm_obs=args['norm_obs'],
        ctx=ctx,
        wrapper_params=args['wrappers'],
        )

    eval_envs = make_vec_env(
            name=args['env_name'], 
            num=args['num_envs'], 
            norm_r=False, 
            norm_obs=args['norm_obs'],
            ctx=ctx,
            wrapper_params=args['wrappers'],)
    eval_envs.training=False


    train_obs_rms = envs.obs_rms
    train_ret_rms = envs.ret_rms

    policy_model = POLICY_CLASSES[args['policy_type']](
        hidden_sizes=args['hidden_sizes'], 
        action_dim=envs.action_space.shape[0],
        init_log_std=args['init_log_std'])

    qf_model = QF_CLASSES[args['q_function']](hidden_sizes=args['q_hidden_sizes'], action_dim=envs.action_space.shape[0],)
    vf_model = VFunction(hidden_sizes=args['q_hidden_sizes'])

    prngkey = jax.random.PRNGKey(args['seed'])

    state = create_train_state(
        prngkey,
        policy_model,
        qf_model,
        vf_model,
        envs,
        learning_rate=args['lr'],
        decaying_lr=args['linear_decay'],
        max_norm=args['max_grad_norm'],
        decay=args['rms_beta2'],
        eps=args['rms_eps'],
        train_steps=num_transition_steps
    )

    _apply_policy_fn = functools.partial(_policy_fn, apply_fn=state.apply_fn, determenistic=False)
    _apply_value_and_policy_fn = functools.partial(
        _value_and_policy_fn, 
        apply_fn=state.apply_fn,
        v_fn=state.v_fn,
        determenistic=False)

    jit_value_and_policy_fn = jax.jit(_apply_value_and_policy_fn)

    next_obs = envs.reset()
    next_obs_and_dones = (next_obs, np.array(next_obs.shape[0]*[False]))


    # ------------------------------------------
    # DATA FROM PREF RUN
    from flax.core import unfreeze, freeze
    import pickle
    import flax

    with open("0.ch", 'rb') as handle:
        additional, good_state_dict = pickle.load(handle)

    original_state_dict = flax.serialization.to_state_dict(state)
    original_state_dict['params']['vf_params'] = good_state_dict['params']['vf_params']

    state = flax.serialization.from_state_dict(state, original_state_dict)

    if args['load_rms']:
        envs.obs_rms = deepcopy(additional['obs_rms'])
        envs.ret_rms = deepcopy(additional['ret_rms'])
        train_obs_rms = envs.obs_rms
        train_ret_rms = envs.ret_rms

    if args['load']:
        chkpnt = args['load']
        if os.path.exists(args['load']):
            print(f"Loading checkpoint {chkpnt}")
            state, additional = load_state(chkpnt, state)
            wandb_run_id = additional['wandb_run_id']
            start_update = state.step

            timestep = state.step * args['num_envs'] * args['num_steps'] 
            envs.obs_rms = deepcopy(additional['obs_rms'])
            envs.ret_rms = deepcopy(additional['ret_rms'])

            train_obs_rms = envs.obs_rms
            train_ret_rms = envs.ret_rms
        else:
            print(f"Checkpoint {chkpnt} not found!")

    if args['wb_flag']:
        if wandb_run_id is None:
            wandb.init(project=args['wandb_proj_name'], config=args)
            wandb_run_id = wandb.run.id
        else:
            wandb.init(project=args['wandb_proj_name'], config=args, id=wandb_run_id, resume="allow")

    total_updates = args['num_timesteps'] // ( args['num_envs'] * args['num_steps'])


    args['train_constants'] = freeze(args['train_constants'])

    jit_q_fn = jax.jit(state.q_fn)

    interactions_per_epoch = calculate_interactions_per_epoch(args)
    for current_update in range(start_update, total_updates):
        st = time.time()
        ready_value_and_policy_fn = functools.partial(
            jit_value_and_policy_fn, 
            params=state.params['policy_params'],
            vf_params=state.params['vf_params'])

        if current_update%args['eval_every']==0:
            eval_envs.obs_rms = deepcopy(envs.obs_rms)
            _, eval_return = eval(state.apply_fn, state.params['policy_params'], eval_envs)
            print(f'Updates {current_update}/{total_updates}. Eval return: {eval_return}. Epoch_time: {epoch_time}.')
        #------------------------------------------------
        #              WORKER ROLLOUTS
        #------------------------------------------------
        
        prngkey, _ = jax.random.split(prngkey)
        next_obs_and_dones, original_experience = collect_experience(
            prngkey, 
            next_obs_and_dones, 
            envs, 
            num_steps=args['num_steps'], 
            policy_fn=ready_value_and_policy_fn,)
        
        # base_oar = process_base_rollout_output(state.apply_fn, state.params, original_experience, args['train_constants'])
        def concat_first_two_dims(x):
            return x.reshape(
                (x.shape[0]*x.shape[1], ) + x.shape[2:]
            )
        observations = concat_first_two_dims(original_experience.observations)
        actions = concat_first_two_dims(original_experience.actions)
        rewards = concat_first_two_dims(original_experience.rewards)
        values = concat_first_two_dims(original_experience.values[-args['num_steps']:])
        returns = rewards + args['gamma']*values
        base_oar = dict(
            observations=observations,
            actions=actions,
            returns=returns,
        )
        prngkey, _ = jax.random.split(prngkey)
        p_train_data_dict = {
            'oar': base_oar,
            'base_oar': base_oar,
            'mc_oar': None,
            'not_sampled_observations': None,
            }

        state, (loss, loss_dict) = p_step(
            state, 
            p_train_data_dict,
            prngkey,
            constant_params=args['train_constants'],
            )
        if args['save'] and (current_update % args['save_every']):
            additional = {}
            additional['obs_rms'] = deepcopy(envs.obs_rms)
            additional['ret_rms'] = deepcopy(envs.ret_rms)
            additional['wandb_run_id'] = wandb_run_id
            save_state(args['save'], state, additional)

        epoch_times.append(time.time() - st)
        epoch_time = np.mean(epoch_times)
        if args['wb_flag'] and not (current_update % args['log_freq']):
            wandb.log({
                'time/timestep': timestep, 
                'time/updates': current_update, 
                'time/time': epoch_time,
                'time/num_interactions': interactions_per_epoch * current_update,
                'evaluation/score': eval_return,
                'evaluation/train_score': envs.get_last_return().mean()}, 
                commit=False, step=current_update)
            epoch_times = []

            loss_dict = jax.tree_map(lambda x: x.item(), loss_dict)
            loss_dict['loss'] = loss.item()
            wandb.log({'training/' + k: v for k, v in loss_dict.items()}, step=current_update)

if __name__=='__main__':

    from args import args

    ctx = mp.get_context("forkserver")

    main(args)

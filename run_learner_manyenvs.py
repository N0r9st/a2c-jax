import functools
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from copy import deepcopy
import time
import jax
import jax.numpy as jnp
import numpy as np
import wandb
import itertools
import multiprocessing as mp

from jax_a2c.a2c import p_step
from jax_a2c.q_updates import q_step, train_test_split, test_qf, train_test_split_k_repeat, general_train_test_split
from jax_a2c.distributions import sample_action_from_normal as sample_action
from jax_a2c.env_utils import make_vec_env, DummySubprocVecEnv, run_workers
from jax_a2c.evaluation import eval, q_eval
from jax_a2c.policy import DiagGaussianPolicy, QFunction, DiagGaussianStateDependentPolicy, VFunction, DGPolicy
from jax_a2c.utils import (Experience, collect_experience, create_train_state, select_random_states,
                           process_experience, concat_trajectories, process_base_rollout_output,
                           stack_experiences, process_rollout_output,  process_mc_rollout_output,
                           calculate_interactions_per_epoch, PRF)
from jax_a2c.km_mc_traj import km_mc_rollouts
from jax_a2c.saving import save_state, load_state
from flax.core import freeze
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from multihost.job_server import KLMJobServer

POLICY_CLASSES = {
    'DiagGaussianPolicy': DiagGaussianPolicy, 
    'DiagGaussianStateDependentPolicy': DiagGaussianStateDependentPolicy,
    "DGPolicy": DGPolicy,
}


def _value_and_policy_fn(prngkey, observation, params, vf_params, apply_fn, v_fn, determenistic=False):
    means, log_stds = apply_fn({'params': params}, observation)
    values = v_fn({'params': vf_params}, observation)
    sampled_actions  = means if determenistic else sample_action(prngkey, means, log_stds)
    return values, sampled_actions


def get_startstates_noad_key(k_envs, num_workers, prngkey: jax.random.PRNGKey, total_parallel):
    out = []
    keys = jax.random.split(prngkey, num=num_workers)
    repeats_per_worker = total_parallel // (k_envs.num_envs * num_workers)
    for i in range(num_workers):
        states_list = []
        next_obs_and_dones_list = []
        
        for j in range(repeats_per_worker):
            next_obs = k_envs.reset()
            next_obs_and_dones = (next_obs, np.array(next_obs.shape[0]*[False]))
            states = k_envs.get_state()

            next_obs_and_dones_list.append(next_obs_and_dones)
            states_list.append(states)
        out.append(
            dict(
                initial_state_list=states_list,
                next_obs_and_dones_list=next_obs_and_dones_list,
                prngkey=keys[i],
                obs_rms=k_envs.obs_rms,
                ret_rms=k_envs.ret_rms
            )
        )
    return out

def main(args: dict):
    if args['type'] != 'standart':
        raise NotImplementedError

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
        num=args['num_k_envs'], 
        norm_r=args['norm_r'], 
        norm_obs=args['norm_obs'],
        ctx=ctx,
        wrapper_params=args['wrappers'],)

    eval_envs = make_vec_env(
            name=args['env_name'], 
            num=16, #args['num_k_envs'], 
            norm_r=False, 
            norm_obs=args['norm_obs'],
            ctx=ctx,
            wrapper_params=args['wrappers'])
    eval_envs.training=False


    train_obs_rms = envs.obs_rms
    train_ret_rms = envs.ret_rms

    policy_model = POLICY_CLASSES[args['policy_type']](
        hidden_sizes=args['hidden_sizes'], 
        action_dim=envs.action_space.shape[0],
        init_log_std=args['init_log_std'])

    qf_model = QFunction(hidden_sizes=args['q_hidden_sizes'], action_dim=envs.action_space.shape[0],)
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
    _apply_value_and_policy_fn = functools.partial(
        _value_and_policy_fn, 
        apply_fn=state.apply_fn,
        v_fn=state.v_fn,
        determenistic=False)

    

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

    # -----------------------------------------
    #            CONNECTING TO REDIS
    #-----------------------------------------
    RESULT_PREFIX = str(args)

    if args['wb_flag']:
        RESULT_PREFIX += wandb_run_id
    server = KLMJobServer(host=args['redis_host'], port=args['redis_port'], password='fuckingpassword',
        base_prefix="manyenvs_")
    print(RESULT_PREFIX)
    server.reset_queue(prefix=RESULT_PREFIX)
    server.reset_queue()
    
    # ------------------------------------------

    total_updates = args['num_timesteps'] // ( args['num_envs'] * args['num_steps'])


    args['train_constants'] = freeze(args['train_constants'])

    interactions_per_epoch = calculate_interactions_per_epoch(args)
    start_collection_data = get_next_obs_and_dones_per_worker(envs, args['num_workers'], args['num_envs'])

    for current_update in range(start_update, total_updates):
        st = time.time()

        if current_update%args['eval_every']==0:
            eval_envs.obs_rms = deepcopy(train_obs_rms)
            _, eval_return = eval(state.apply_fn, state.params['policy_params'], eval_envs)
            print(f'Updates {current_update}/{total_updates}. Eval return: {eval_return}. Epoch_time: {epoch_time}.')
        #------------------------------------------------
        #              WORKER ROLLOUTS
        #------------------------------------------------
    
        exp_list = []
        not_selected_observations_list = []

        with PRF("SENDING JOBS"):
            # for starter_info_lists in states_and_noad:
            #     starter_info_lists.update(obs_rms=train_obs_rms, ret_rms=train_ret_rms,)
            #     starter_info_lists.update(num_steps=args['num_steps'],)
            #     to_worker = dict(
            #         policy_params=state.params['policy_params'],
            #         vf_params=state.params['vf_params'],
            #         args=starter_info_lists,
            #         iteration=current_update,
            #         prefix=RESULT_PREFIX,
            #         )
            #     server.add_jobs(to_worker)
            #     print(current_update, '- ADDED JOB')

            for next_obs_and_dones, start_states in start_collection_data:
                collect_experience_args = dict(
                    prngkey=prngkey,
                    next_obs_and_dones=next_obs_and_dones,
                    start_states=start_states,
                    num_steps=args['num_envs'],
                )
                to_worker = dict(
                    policy_params=state.params['policy_params'],
                    vf_params=state.params['vf_params'],
                    collect_experience_args=collect_experience_args,
                    obs_rms=train_obs_rms,
                    ret_rms=train_ret_rms,
                    iteration=current_update,
                    prefix=RESULT_PREFIX,
                    )
                server.add_jobs(to_worker)
                print(current_update, '- ADDED JOB')

        with PRF("WAITING RESULTS"):
            list_dict_results, workers_logs  = server.get_job_results(
                current_update, 
                args['n_packages'],
                negative=False,
                prefix=RESULT_PREFIX,
                )

        with PRF("PREPROCESSING RESULTS"):
            states_and_noad = []
            exp_list = []
            for dict_ in list_dict_results:
                starter_info, exp_w_list = dict_['experiences']
                exp_list += exp_w_list
                states_and_noad.append(starter_info)
            train_obs_rms = states_and_noad[0]['obs_rms']
            train_ret_rms = states_and_noad[0]['ret_rms']

            original_experience = stack_experiences(exp_list)


            base_oar = process_base_rollout_output(state.apply_fn, state.params, original_experience, args['train_constants'])

        negative_oar = None
        mc_oar = None
        oar = base_oar
        not_sampled_observations = base_oar['observations'].reshape((-1, base_oar['observations'].shape[-1]))
        sampling_masks = None
        no_sampling_masks = None

        prngkey, _ = jax.random.split(prngkey)
        p_train_data_dict = {
            'oar': oar,
            'base_oar': base_oar,
            'mc_oar': mc_oar,
            'not_sampled_observations': not_sampled_observations,
            }
        with PRF("POLICY STEP"):
            state, (loss, loss_dict) = p_step(
                state, 
                p_train_data_dict,
                prngkey,
                constant_params=args['train_constants'],
                )
        if args['save'] and (current_update % args['save_every']):
            additional = {}
            additional['obs_rms'] = deepcopy(train_obs_rms)
            additional['ret_rms'] = deepcopy(train_ret_rms)
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
            wandb.log({'multihost/' + k: v for k, v in workers_logs.items()}, step=current_update)


if __name__=='__main__':

    from args import args

    ctx = mp.get_context("forkserver")

    main(args)

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
from jax_a2c.policy import (DiagGaussianPolicy,
                            DiagGaussianStateDependentPolicy, QFunction)
from jax_a2c.q_updates import (general_train_test_split, groub_by_repeats,
                               q_step, test_qf, train_test_split,
                               train_test_split_k_repeat)
from jax_a2c.saving import load_state, save_state
from jax_a2c.utils import (Experience, calculate_interactions_per_epoch,
                           collect_experience, concat_trajectories,
                           create_train_state, process_base_rollout_output,
                           process_experience, process_mc_rollout_output,
                           process_rollout_output, select_random_states,
                           stack_experiences)

POLICY_CLASSES = {
    'DiagGaussianPolicy': DiagGaussianPolicy, 
    'DiagGaussianStateDependentPolicy': DiagGaussianStateDependentPolicy,
}
def _policy_fn(prngkey, observation, params, apply_fn, determenistic=False):
    values, (means, log_stds) = apply_fn({'params': params}, observation)
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

    while True:
        try:
            args = remote.recv()
            k_envs.obs_rms = args.pop('train_obs_rms')
            k_envs.ret_rms = args.pop('train_ret_rms')
            policy_fn = functools.partial(_policy_fn, **(args.pop('policy_fn')))
            out = km_mc_rollouts_(policy_fn=policy_fn, **args)
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
        ctx=ctx)
        
    k_envs_fn = functools.partial(make_vec_env,
        name=args['env_name'], 
        num=args['num_k_envs'], 
        norm_r=args['norm_r'], 
        norm_obs=args['norm_obs'],
        ctx=ctx
        )

    eval_envs = make_vec_env(
            name=args['env_name'], 
            num=args['num_envs'], 
            norm_r=False, 
            norm_obs=args['norm_obs'],
            ctx=ctx)
    eval_envs.training=False


    train_obs_rms = envs.obs_rms
    train_ret_rms = envs.ret_rms

    policy_model = POLICY_CLASSES[args['policy_type']](
        hidden_sizes=args['hidden_sizes'], 
        action_dim=envs.action_space.shape[0],
        init_log_std=args['init_log_std'])

    qf_model = QFunction(hidden_sizes=args['q_hidden_sizes'], action_dim=envs.action_space.shape[0],)

    prngkey = jax.random.PRNGKey(args['seed'])

    state = create_train_state(
        prngkey,
        policy_model,
        qf_model,
        envs,
        learning_rate=args['lr'],
        decaying_lr=args['linear_decay'],
        max_norm=args['max_grad_norm'],
        decay=args['rms_beta2'],
        eps=args['rms_eps'],
        train_steps=num_transition_steps
    )

    _apply_policy_fn = functools.partial(_policy_fn, apply_fn=state.apply_fn, determenistic=False)
    _jit_policy_fn = jax.jit(_apply_policy_fn)

    next_obs = envs.reset()
    next_obs_and_dones = (next_obs, np.array(next_obs.shape[0]*[False]))


    # -----------------------------------------
    #            STARTING WORKERS
    #-----------------------------------------
    if args['type'] != 'standart':
        if args['num_workers'] is not None:
            add_args = {'policy_fn': _apply_policy_fn}
            remotes = run_workers(
                _worker, 
                k_envs_fn, 
                args['num_workers'], 
                (envs.observation_space, envs.action_space),
                ctx,
                split_between_devices=args['split_between_devices'],
                add_args=add_args)
    # ------------------------------------------

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
        policy_fn = functools.partial(_jit_policy_fn, params=state.params['policy_params'])
        if current_update%args['eval_every']==0:
            eval_envs.obs_rms = deepcopy(envs.obs_rms)
            _, eval_return = eval(state.apply_fn, state.params['policy_params'], eval_envs)
            print(f'Updates {current_update}/{total_updates}. Eval return: {eval_return}. Epoch_time: {epoch_time}.')

            if args['eval_with_q']:
                eval_envs.obs_rms = deepcopy(envs.obs_rms)
                _, q_eval_return = q_eval(state.apply_fn, state.params['policy_params'], 
                                        state.q_fn, state.params['qf_params'], eval_envs)
                print(f'Q-eval return: {q_eval_return}')

        #------------------------------------------------
        #              WORKER ROLLOUTS
        #------------------------------------------------
        if args['type'] != 'standart':
            exp_list = []
            add_args_list = []
            not_selected_observations_list = []
            sampled_exp_list = []
            sampling_masks = []
            no_sampling_masks = []

            for remote in remotes:

                prngkey, _ = jax.random.split(prngkey)
                next_obs_and_dones, experience = collect_experience(
                    prngkey, 
                    next_obs_and_dones, 
                    envs, 
                    num_steps=args['num_steps']//args['num_workers'], 
                    policy_fn=policy_fn,)
                exp_list.append(experience)


                
                add_args = {}
                if args['sampling_type']=='adv':
                    base_traj_part = process_experience(experience, gamma=args['gamma'], lambda_=args['lambda_'])
                    advs = base_traj_part[3].reshape((args['num_steps']//args['num_workers'], args['num_envs']))
                    add_args['advantages'] = advs
                    add_args['sampling_prob_temp'] = args['sampling_prob_temp']
                    
                sampled_exp, not_selected_observations, sampling_mask, no_sampling_mask = select_random_states(
                    prngkey, args['n_samples']//args['num_workers'], 
                    experience, type=args['sampling_type'], **add_args)

                sampling_masks.append(sampling_mask)
                no_sampling_masks.append(no_sampling_mask)
                not_selected_observations_list.append(not_selected_observations)
                sampled_exp_list.append(sampled_exp)

                add_args_list.append(add_args)
                sampled_exp = Experience(
                    observations=sampled_exp.observations,
                    actions=None,
                    rewards=None,
                    values=None,
                    dones=sampled_exp.dones,
                    states=sampled_exp.states,
                    next_observations= None,
                )
                prngkey, _ = jax.random.split(prngkey)
                to_worker = dict(
                    prngkey=prngkey,
                    experience=sampled_exp,
                    gamma=args['gamma'],
                    policy_fn=dict(
                        params=state.params['policy_params'], 
                        determenistic=args['km_determenistic']),
                    max_steps=args['L'],
                    K=args['K'],
                    M=args['M'],
                    train_obs_rms=train_obs_rms,
                    train_ret_rms=train_ret_rms,
                    firstrandom=False,
                    )
                remote.send(to_worker)

            original_experience = stack_experiences(exp_list)
            mc_experience =  jax.tree_util.tree_map(
                lambda *dicts: jnp.stack(dicts),
                *[remote.recv() for remote in remotes],
                )

            #----------------------------------------------------------------
            #                       NEGATIVES SAMPLING
            #----------------------------------------------------------------
            negative_oar = None
            if args['negative_sampling']:
                for remote, add_args, sampled_exp in zip(remotes, add_args_list, sampled_exp_list):
                    prngkey, _ = jax.random.split(prngkey)
                    to_worker = dict(
                        prngkey=prngkey,
                        experience=sampled_exp,
                        gamma=args['gamma'],
                        policy_fn=dict(
                            params=state.params['policy_params'], 
                            determenistic=args['km_determenistic']),
                        max_steps=args['L'],
                        K=args['K'],
                        M=args['M'],
                        train_obs_rms=train_obs_rms,
                        train_ret_rms=train_ret_rms,
                        firstrandom=True,
                        )
                    remote.send(to_worker)
                negative_exp = jax.tree_util.tree_map(lambda *dicts: jnp.stack(dicts),
                    *[remote.recv() for remote in remotes]
                    )
                negative_oar = process_mc_rollout_output(
                    state.apply_fn, state.params, 
                    negative_exp, args['train_constants'])
            #----------------------------------------------------------------
            #----------------------------------------------------------------
        else:
            prngkey, _ = jax.random.split(prngkey)
            next_obs_and_dones, original_experience = collect_experience(
                prngkey, 
                next_obs_and_dones, 
                envs, 
                num_steps=args['num_steps'], 
                policy_fn=policy_fn,)
        
        base_oar = process_base_rollout_output(state.apply_fn, state.params, original_experience, args['train_constants'])

        if args['type'] != 'standart':
            mc_oar = process_mc_rollout_output(state.apply_fn, state.params, mc_experience, args['train_constants'])
            oar = dict(
                observations=jnp.concatenate((base_oar['observations'], mc_oar['observations']), axis=0),
                actions=jnp.concatenate((base_oar['actions'], mc_oar['actions']), axis=0),
                returns=jnp.concatenate((base_oar['returns'], mc_oar['returns']), axis=0),
                )
            not_sampled_observations = jnp.concatenate(not_selected_observations_list, axis=0)
        else:
            negative_oar = None
            mc_oar = None
            oar = base_oar
            not_sampled_observations = base_oar['observations'].reshape((-1, base_oar['observations'].shape[-1]))

        prngkey, _ = jax.random.split(prngkey)
        for n_samples_slice in [16, 32, 64, 128, 256]:
            for base_traj in [False, True]:
                mc_oar_ = groub_by_repeats(mc_oar, args['K'], args['num_workers'])

                def _slice_grouped_mc_oar(arr):
                    arr = arr[:, :n_samples_slice, ...]
                    return arr.reshape((arr.shape[0]*arr.shape[1],) + arr.shape[2:])
                mc_oar_ = jax.tree_util.tree_map(_slice_grouped_mc_oar, mc_oar_)

                q_train_oar, q_test_oar = general_train_test_split(
                    base_oar=base_oar,
                    mc_oar=mc_oar_,
                    negative_oar=negative_oar,
                    prngkey=prngkey,
                    test_ratio=args['train_constants']['qf_test_ratio'],
                    k=args['K'],
                    nw=args['num_workers'],
                    use_base_traj_for_q=args['use_base_traj_for_q'],
                    full_tt_split=args['full_tt_split'],
                )

                args['train_constants'] = args['train_constants'].copy({
                    'qf_update_batch_size':args['train_constants']['qf_update_batch_size'],
                    'q_train_len':len(q_train_oar['observations']),
                    })

                state_qupdated, (q_loss, q_loss_dict) = q_step(
                    state, 
                    # trajectories, 
                    q_train_oar, q_test_oar, # (Experience(original trajectory), List[dicts](kml trajs))
                    prngkey,
                    constant_params=args['train_constants'], jit_q_fn=jit_q_fn
                    )
                state_qupdated = state_qupdated.replace(step=current_update)
                wandb.log({f'q-training-{n_samples_slice}-b{base_traj}/' + k: v for k, v in q_loss_dict.items()}, step=current_update)

        state = state_qupdated
        prngkey, _ = jax.random.split(prngkey)
        p_train_data_dict = {
            'oar': oar,
            'not_sampled_observations': not_sampled_observations,
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
                'time/num_interactions': interactions_per_epoch,
                'evaluation/score': eval_return,
                'evaluation/train_score': envs.get_last_return().mean()}, 
                commit=False, step=current_update)
            epoch_times = []

            loss_dict = jax.tree_map(lambda x: x.item(), loss_dict)
            q_loss_dict = jax.tree_map(lambda x: x.item(), q_loss_dict)
            loss_dict['loss'] = loss.item()
            wandb.log({'training/' + k: v for k, v in loss_dict.items()}, step=current_update)
            wandb.log({'q-training/' + k: v for k, v in q_loss_dict.items()}, step=current_update)

if __name__=='__main__':

    from args import args

    ctx = mp.get_context("forkserver")

    main(args)

import functools
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from copy import deepcopy
import time
import jax
import numpy as np
import wandb
import itertools
import multiprocessing as mp

from jax_a2c.a2c import step
from jax_a2c.distributions import sample_action_from_normal as sample_action
from jax_a2c.env_utils import make_vec_env, DummySubprocVecEnv, run_workers
from jax_a2c.evaluation import eval
from jax_a2c.policy import DiagGaussianPolicy
from jax_a2c.utils import (Experience, collect_experience, create_train_state,
                           process_experience, concat_trajectories, stack_experiences, select_random_states)
from jax_a2c.km_mc_traj import km_mc_rollouts_trajectories
from jax_a2c.saving import save_state, load_state
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

def _policy_fn(prngkey, observation, params, apply_fn, det=False):
    values, (means, log_stds) = apply_fn({'params': params}, observation)
    sampled_actions  = means if det else sample_action(prngkey, means, log_stds)
    return values, sampled_actions

def _worker(remote, k_remotes, parent_remote, spaces, device) -> None:
    print('D:', device)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    parent_remote.close()
    k_envs = DummySubprocVecEnv(remotes=k_remotes)
    
    k_envs.observation_space, k_envs.action_space = spaces
    k_envs = VecNormalize(k_envs, training=False)
    km_mc_rollouts_trajectories_ = functools.partial(km_mc_rollouts_trajectories, k_envs=k_envs)
    while True:
        try:
            args = remote.recv()
            k_envs.obs_rms = args.pop('train_obs_rms')
            k_envs.ret_rms = args.pop('train_ret_rms')
            args['policy_fn'] = jax.jit(args['policy_fn'])
            out = km_mc_rollouts_trajectories_(**args)
            remote.send(out)
        except EOFError:
            break

def main(args: dict):
    args['async'] = True
    if not args['split_between_devices']:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
    # os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = args['allocate_memory']

    num_transition_steps = args['num_timesteps']//(args['num_envs'] * args['num_steps'])
    wandb_run_id = None
    start_update = 0
    timestep = 0
    epoch_times = []
    epoch_time = 0
    eval_return = None

    envs = make_vec_env(
        name=args['env_name'], 
        num=args['num_envs'], 
        norm_r=args['norm_r'], 
        norm_obs=args['norm_obs'],
        ctx=ctx)

    if args['type'] in ["KM-rollouts", "sample-KM-rollouts", 'sample-KM-rollouts-fast']:
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

    # -----------------------------------------
    #            STARTING WORKERS
    #-----------------------------------------
    if args['type'] in ["KM-rollouts", "sample-KM-rollouts", 'sample-KM-rollouts-fast']:
        if args['num_workers'] is not None:
            remotes = run_workers(
                _worker, 
                k_envs_fn, 
                args['num_workers'], 
                (envs.observation_space, envs.action_space),
                ctx,
                split_between_devices=args['split_between_devices'])
    # ------------------------------------------

    model = DiagGaussianPolicy(
        hidden_sizes=args['hidden_sizes'], 
        action_dim=envs.action_space.shape[0],
        init_log_std=args['init_log_std'])

    prngkey = jax.random.PRNGKey(args['seed'])

    state = create_train_state(
        prngkey,
        model,
        envs,
        learning_rate=args['lr'],
        decaying_lr=args['linear_decay'],
        max_norm=args['max_grad_norm'],
        decay=args['rms_beta2'],
        eps=args['rms_eps'],
        train_steps=num_transition_steps
    )

    _apply_policy_fn = functools.partial(_policy_fn, apply_fn=state.apply_fn)
    _jit_policy_fn = jax.jit(
        functools.partial(_apply_policy_fn, det=False),
        )

    next_obs = envs.reset()
    next_obs_and_dones = (next_obs, np.array(next_obs.shape[0]*[False]))

    if args['load']:
        chkpnt = args['load']
        if os.path.exists(args['load']):
            print(f"Loading checkpoint {chkpnt}")
            state, additional = load_state(chkpnt, state)
            wandb_run_id = additional['wandb_run_id']
            start_update = state.step

            timestep = state.step * args['num_envs'] * args['num_steps'] # state.step * args['K'] * args['L'] * args['M'] * args['num_envs'] * args['num_steps'] 

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


    for current_update in range(start_update, total_updates):
        st = time.time()
        policy_fn = functools.partial(_jit_policy_fn, params=state.params)
        if state.step%args['eval_every']==0:
            eval_envs.obs_rms = deepcopy(envs.obs_rms)
            _, eval_return = eval(state.apply_fn, state.params, eval_envs)
            print(f'Updates {current_update}/{total_updates}. Eval return: {eval_return}. Epoch_time: {epoch_time}.')

        #------------------------------------------------
        #              WORKER ROLLOUTS
        #------------------------------------------------
        if args['type'] == "KM-rollouts":
            exp_list = []
            for remote in remotes:
                prngkey, _ = jax.random.split(prngkey)
                next_obs_and_dones, experience = collect_experience(
                    prngkey, 
                    next_obs_and_dones, 
                    envs, 
                    num_steps=args['num_steps']//args['num_workers'], 
                    policy_fn=policy_fn,)
                exp_list.append(experience)
                
                prngkey, _ = jax.random.split(prngkey)
                to_worker = dict(
                    prngkey=prngkey,
                    experience=experience,
                    gamma=args['gamma'],
                    policy_fn=functools.partial(_apply_policy_fn, params=state.params, det=args['det_km']),
                    max_steps=args['L'],
                    K=args['K'],
                    M=args['M'],
                    train_obs_rms=train_obs_rms,
                    train_ret_rms=train_ret_rms,)
                remote.send(to_worker)
            base_traj = process_experience(stack_experiences(exp_list), gamma=args['gamma'], lambda_=args['lambda_'])
            trajectories = concat_trajectories([remote.recv() for remote in remotes] + \
                [base_traj]*(1-args['ignore_original_trajectory']))
        # --------------------------------------------

        if args['type'] == 'sample-KM-rollouts':
            prngkey, _ = jax.random.split(prngkey)
            next_obs_and_dones, experience = collect_experience(
                prngkey, 
                next_obs_and_dones, 
                envs, 
                num_steps=args['num_steps'], 
                policy_fn=policy_fn,)

            base_traj = process_experience(experience, gamma=args['gamma'], lambda_=args['lambda_'])
            if args['sampling_type']=='adv':
                advs = base_traj[3].reshape((args['num_steps'], args['num_envs']))
            
            
            add_args = {}
            if args['sampling_type']=='adv':
                add_args['advantages'] = advs
                add_args['sampling_prob_temp'] = args['sampling_prob_temp']

            sampled_exp = select_random_states(prngkey, args['n_samples'], experience, type=args['sampling_type'], **add_args)
            send_amount = args['n_samples']//args['num_workers']

            for i_r, remote in enumerate(remotes):
                
                exp = Experience(
                    observations=sampled_exp.observations[i_r*send_amount: (i_r+1)*send_amount][None],
                    actions=sampled_exp.actions[i_r*send_amount: (i_r+1)*send_amount][None],
                    rewards=sampled_exp.rewards[i_r*send_amount: (i_r+1)*send_amount][None],
                    values=sampled_exp.values[i_r*send_amount: (i_r+1)*send_amount][None],
                    dones=sampled_exp.dones[i_r*send_amount: (i_r+1)*send_amount][None],
                    states=[sampled_exp.states[i_r*send_amount: (i_r+1)*send_amount]],
                )

                prngkey, _ = jax.random.split(prngkey)
                to_worker = dict(
                    prngkey=prngkey,
                    experience=exp,
                    gamma=args['gamma'],
                    policy_fn=functools.partial(_apply_policy_fn, params=state.params, det=args['det_km']),
                    max_steps=args['L'],
                    K=args['K'],
                    M=args['M'],
                    train_obs_rms=train_obs_rms,
                    train_ret_rms=train_ret_rms,)
                remote.send(to_worker)
            trajectories = concat_trajectories([remote.recv() for remote in remotes] \
                + [base_traj]*(1-args['ignore_original_trajectory']))

        if args['type'] == 'sample-KM-rollouts-fast':
            exp_list = []
            for remote in remotes:

                prngkey, _ = jax.random.split(prngkey)
                next_obs_and_dones, experience = collect_experience(
                    prngkey, 
                    next_obs_and_dones, 
                    envs, 
                    num_steps=args['num_steps']//args['num_workers'], 
                    policy_fn=policy_fn,)
                exp_list.append(experience)

                base_traj_part = process_experience(experience, gamma=args['gamma'], lambda_=args['lambda_'])
                add_args = {}
                if args['sampling_type']=='adv':
                    advs = base_traj_part[3].reshape((args['num_steps']//args['num_workers'], args['num_envs']))
                    add_args['advantages'] = advs
                    add_args['sampling_prob_temp'] = args['sampling_prob_temp']
                sampled_exp = select_random_states(prngkey, args['n_samples']//args['num_workers'], experience, type=args['sampling_type'], **add_args)
                sampled_exp = Experience(
                    observations=sampled_exp.observations[None],
                    actions=sampled_exp.actions[None],
                    rewards=sampled_exp.rewards[None],
                    values=sampled_exp.values[None],
                    dones=sampled_exp.dones[None],
                    states=[sampled_exp.states],
                )

                prngkey, _ = jax.random.split(prngkey)
                to_worker = dict(
                    prngkey=prngkey,
                    experience=sampled_exp,
                    gamma=args['gamma'],
                    policy_fn=functools.partial(_apply_policy_fn, params=state.params, det=args['det_km']),
                    max_steps=args['L'],
                    K=args['K'],
                    M=args['M'],
                    train_obs_rms=train_obs_rms,
                    train_ret_rms=train_ret_rms,
                    )
                remote.send(to_worker)
            base_traj = process_experience(stack_experiences(exp_list), gamma=args['gamma'], lambda_=args['lambda_'])
            trajectories = concat_trajectories([remote.recv() for remote in remotes] \
                + [base_traj]*(1-args['ignore_original_trajectory']))

        elif args['type'] == 'standart':
            prngkey, _ = jax.random.split(prngkey)
            next_obs_and_dones, experience = collect_experience(
                prngkey, 
                next_obs_and_dones, 
                envs, 
                num_steps=args['num_steps'], 
                policy_fn=policy_fn,)
            trajectories = process_experience(experience, gamma=args['gamma'], lambda_=args['lambda_'])
        #----------------------------------------------------------------

        timestep += len(trajectories[0])
            
        for _ in range(args['updates_per_batch']):
            state, (loss, loss_dict) = step(
                state, 
                trajectories, 
                value_loss_coef=args['value_loss_coef'], 
                entropy_coef=args['entropy_coef'], 

                normalize_advantages=args['normalize_advantages'])

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
                'evaluation/score': eval_return}, 
                commit=False, step=current_update)
            epoch_times = []

            loss_dict = jax.tree_map(lambda x: x.item(), loss_dict)
            loss_dict['loss'] = loss.item()
            wandb.log({'training/' + k: v for k, v in loss_dict.items()}, step=current_update)

if __name__=='__main__':

    from args import args

    ctx = mp.get_context("forkserver")

    main(args)

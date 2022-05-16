from ast import arg
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
from jax_a2c.q_updates import q_step, train_test_split
from jax_a2c.distributions import sample_action_from_normal as sample_action
from jax_a2c.env_utils import make_vec_env, DummySubprocVecEnv, run_workers
from jax_a2c.evaluation import eval, q_eval
from jax_a2c.policy import DiagGaussianPolicy, QFunction, DiagGaussianStateDependentPolicy
from jax_a2c.utils import (Experience, collect_experience, create_train_state, select_random_states,
                           process_experience, concat_trajectories, 
                           stack_experiences, process_rollout_output,  process_mc_rollout_output)
from jax_a2c.km_mc_traj import km_mc_rollouts
from jax_a2c.saving import save_state, load_state
from flax.core import freeze
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize

POLICY_CLASSES = {
    'DiagGaussianPolicy': DiagGaussianPolicy, 
    'DiagGaussianStateDependentPolicy': DiagGaussianStateDependentPolicy,
}
def _policy_fn(prngkey, observation, params, apply_fn, determenistic=False):
    values, (means, log_stds) = apply_fn({'params': params}, observation)
    sampled_actions  = means if determenistic else sample_action(prngkey, means, log_stds)
    return values, sampled_actions

def _worker(remote, k_remotes, parent_remote, spaces, device) -> None:
    print('D:', device)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    parent_remote.close()
    k_envs = DummySubprocVecEnv(remotes=k_remotes)
    
    k_envs.observation_space, k_envs.action_space = spaces
    k_envs = VecNormalize(k_envs, training=False)
    km_mc_rollouts_ = functools.partial(km_mc_rollouts, k_envs=k_envs)
    while True:
        try:
            args = remote.recv()
            k_envs.obs_rms = args.pop('train_obs_rms')
            k_envs.ret_rms = args.pop('train_ret_rms')
            args['policy_fn'] = jax.jit(args['policy_fn'])
            out = km_mc_rollouts_(**args)
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

    
    # -----------------------------------------
    #            STARTING WORKERS
    #-----------------------------------------
    if args['type'] != 'standart':
        if args['num_workers'] is not None:
            remotes = run_workers(
                _worker, 
                k_envs_fn, 
                args['num_workers'], 
                (envs.observation_space, envs.action_space),
                ctx,
                split_between_devices=args['split_between_devices'])
    # ------------------------------------------

    policy_model = POLICY_CLASSES[args['policy_type']](
        hidden_sizes=args['hidden_sizes'], 
        action_dim=envs.action_space.shape[0],
        init_log_std=args['init_log_std'])

    qf_model = QFunction(hidden_sizes=args['hidden_sizes'], action_dim=envs.action_space.shape[0],)

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
    for current_update in range(start_update, total_updates):
        st = time.time()
        policy_fn = functools.partial(_jit_policy_fn, params=state.params['policy_params'])
        if state.step%args['eval_every']==0:
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
            not_sampled_exp_list = []
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
                    
                sampled_exp, not_sampled_exp = select_random_states(
                    prngkey, args['n_samples']//args['num_workers'], 
                    experience, type=args['sampling_type'], **add_args)

                not_sampled_exp_list.append(not_sampled_exp)

                add_args_list.append(add_args)
                sampled_exp = Experience(
                    observations=sampled_exp.observations,
                    actions=None,# sampled_exp.actions,
                    rewards=None,# sampled_exp.rewards,
                    values=None,# sampled_exp.values,
                    dones=sampled_exp.dones,
                    states=sampled_exp.states,
                    next_observations= None,
                )
                prngkey, _ = jax.random.split(prngkey)
                to_worker = dict(
                    prngkey=prngkey,
                    experience=sampled_exp,
                    gamma=args['gamma'],
                    policy_fn=functools.partial(
                        _apply_policy_fn, 
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
            # not_sampled_exp = stack_experiences(not_sampled_exp_list)
            data_tuple = (
                original_experience, 
                jax.tree_util.tree_map(lambda *dicts: jnp.stack(dicts),
                    *[remote.recv() for remote in remotes]
                    )
                )

            #----------------------------------------------------------------
            #                       NEGATIVES SAMPLING
            #----------------------------------------------------------------
            
            if args['negative_sampling']:
                for remote, experience, add_args in zip(remotes, exp_list, add_args_list):
                    prngkey, _ = jax.random.split(prngkey)
                        
                    sampled_exp, not_sampled_exp= select_random_states(
                        prngkey, 
                        args['n_samples']//args['num_workers'], 
                        experience, type=args['sampling_type'], **add_args)
                    prngkey, _ = jax.random.split(prngkey)
                    to_worker = dict(
                        prngkey=prngkey,
                        experience=sampled_exp,
                        gamma=args['gamma'],
                        policy_fn=functools.partial(
                            _apply_policy_fn, 
                            params=state.params['policy_params'], 
                            determenistic=args['km_determenistic']),
                        max_steps=args['L'],
                        K=args['K'],
                        M=args['M'],
                        train_obs_rms=train_obs_rms,
                        train_ret_rms=train_ret_rms,
                        firstrandom=True,
                        )
                    # print('sent!')
                    remote.send(to_worker)
                negative_exp = jax.tree_util.tree_map(lambda *dicts: jnp.stack(dicts),
                    *[remote.recv() for remote in remotes]
                    )
                # print('received')
                negative_oar = process_mc_rollout_output(
                    state.apply_fn, state.params, 
                    negative_exp, args['train_constants'])
            #----------------------------------------------------------------
            #----------------------------------------------------------------
        else:
            prngkey, _ = jax.random.split(prngkey)
            next_obs_and_dones, experience = collect_experience(
                prngkey, 
                next_obs_and_dones, 
                envs, 
                num_steps=args['num_steps'], 
                policy_fn=policy_fn,)
            data_tuple = (
                experience, 
                None,
                )
        
        oar = process_rollout_output(state.apply_fn, state.params, data_tuple, args['train_constants'])
        prngkey, _ = jax.random.split(prngkey)

        args['train_constants'] = args['train_constants'].copy({
            # 'num_train_samples':len(oar['observations']),
            'qf_update_batch_size':args['train_constants']['qf_update_batch_size'] if \
            args['train_constants']['qf_update_batch_size'] > 0 else len(oar['observations']) 
            })
        if args['negative_sampling']:
            q_train_oar, q_test_oar = train_test_split(
                jax.tree_util.tree_map(lambda *dicts: jnp.concatenate(dicts, axis=0),*(oar, negative_oar)),
                prngkey, 
                args['train_constants']['qf_test_ratio'], 
                len(oar['observations']) + len(negative_oar['observations']))

        else:
            q_train_oar, q_test_oar = train_test_split(
                oar,
                prngkey, 
                args['train_constants']['qf_test_ratio'], 
                len(oar['observations']))

        state, (q_loss, q_loss_dict) = q_step(
            state, 
            # trajectories, 
            (q_train_oar, q_test_oar), # (Experience(original trajectory), List[dicts](kml trajs))
            prngkey,
            constant_params=args['train_constants'],
            )
        prngkey, _ = jax.random.split(prngkey)
        state, (loss, loss_dict) = p_step(
            state, 
            # trajectories, 
            oar, # (Experience(original trajectory), List[dicts](kml trajs))
            prngkey,
            constant_params=args['train_constants'],
            )
        loss_dict.update(q_loss_dict)
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
                'evaluation/score': eval_return,}, 
                commit=False, step=current_update)
            epoch_times = []

            loss_dict = jax.tree_map(lambda x: x.item(), loss_dict)
            loss_dict['loss'] = loss.item()
            wandb.log({'training/' + k: v for k, v in loss_dict.items()}, step=current_update)

if __name__=='__main__':

    from args import args

    ctx = mp.get_context("forkserver")

    main(args)

import functools
import os

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import multiprocessing as mp
import time
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze

import wandb
from jax_a2c.a2c import p_step
from jax_a2c.distributions import sample_action_from_normal as sample_action
from jax_a2c.env_utils import make_vec_env
from jax_a2c.evaluation import eval
from jax_a2c.policy import (DiagGaussianPolicy,
                            DiagGaussianStateDependentPolicy, QFunction, VFunction, DGPolicy)
from jax_a2c.q_updates import general_train_test_split, q_step
from jax_a2c.saving import load_state, save_state
from jax_a2c.utils import (Experience, calculate_interactions_per_epoch,
                           collect_experience, create_train_state,
                           process_base_rollout_output, process_experience,
                           select_random_states, stack_experiences)
from multihost.job_server import KLMJobServer

POLICY_CLASSES = {
    'DiagGaussianPolicy': DiagGaussianPolicy, 
    'DiagGaussianStateDependentPolicy': DiagGaussianStateDependentPolicy,
    "DGPolicy": DGPolicy,
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

def main(args: dict):
    RESULT_PREFIX = str(args)
    if args['type']!='sample-KM-rollouts-fast':
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

    envs = make_vec_env(
        name=args['env_name'], 
        num=args['num_envs'], 
        norm_r=args['norm_r'], 
        norm_obs=args['norm_obs'],
        ctx=ctx)

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

    jit_value_and_policy_fn = jax.jit(_apply_value_and_policy_fn)

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

    # -----------------------------------------
    #            CONNECTING TO REDIS
    #-----------------------------------------
    RESULT_PREFIX = str(args)

    if args['wb_flag']:
        RESULT_PREFIX += wandb_run_id
    server = KLMJobServer(host=args['redis_host'], port=args['redis_port'], password='fuckingpassword')
    print(RESULT_PREFIX)
    server.reset_queue(prefix=RESULT_PREFIX)
    
    # ------------------------------------------

    total_updates = args['num_timesteps'] // ( args['num_envs'] * args['num_steps'])


    args['train_constants'] = freeze(args['train_constants'])

    jit_q_fn = jax.jit(state.q_fn)

    interactions_per_epoch = calculate_interactions_per_epoch(args)
    for current_update in range(start_update, total_updates):
        # time_base_collection_and_sending = 0
        time_base_collection = 0
        time_sending = 0
        time_selecting = 0


        time_total_rollouts = 0
        time_q_updates = 0
        time_policy_updates = 0
        time_evaluation = 0

        st = time.time()
        ready_value_and_policy_fn = functools.partial(
            jit_value_and_policy_fn, 
            params=state.params['policy_params'],
            vf_params=state.params['vf_params'])
        
        _st = time.time()
        if current_update%args['eval_every']==0:
            eval_envs.obs_rms = deepcopy(envs.obs_rms)
            _, eval_return = eval(state.apply_fn, state.params['policy_params'], eval_envs)
            print(f'Updates {current_update}/{total_updates}. Eval return: {eval_return}. Epoch_time: {epoch_time}.')
        time_evaluation = time.time() - _st
        #------------------------------------------------
        #              WORKER ROLLOUTS
        #------------------------------------------------
        exp_list = []
        add_args_list = []
        not_selected_observations_list = []
        sampled_exp_list = []
        sampling_masks = []
        no_sampling_masks = []

        # _st = time.time()
        for npgk in range(args['n_packages']):
            _st = time.time()
            prngkey, _ = jax.random.split(prngkey)
            next_obs_and_dones, experience = collect_experience(
                prngkey, 
                next_obs_and_dones, 
                envs, 
                num_steps=args['num_steps']//args['n_packages'], 
                policy_fn=ready_value_and_policy_fn,)
            exp_list.append(experience)

            time_base_collection += time.time() - _st

            _st = time.time()
            add_args = {}
            if args['sampling_type']=='adv':
                base_traj_part = process_experience(experience, gamma=args['gamma'], lambda_=args['lambda_'])
                advs = base_traj_part[3].reshape((args['num_steps']//args['num_workers'], args['num_envs']))
                add_args['advantages'] = advs
                add_args['sampling_prob_temp'] = args['sampling_prob_temp']
            sampled_exp, not_selected_observations, sampling_mask, no_sampling_mask = select_random_states(
                prngkey, args['n_samples']//args['n_packages'], 
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
                    # v_fn=dict(params=state.params["vf_params"]),
                    vf_params=state.params["vf_params"],
                    max_steps=args['L'],
                    K=args['K'],
                    M=args['M'],
                    train_obs_rms=train_obs_rms,
                    train_ret_rms=train_ret_rms,
                    firstrandom=False,
                    iteration=current_update,
                    prefix=RESULT_PREFIX,
                    )

            time_selecting += time.time() - _st
            # remote.send(to_worker)

            _st = time.time()
            print("ADDING JOB")
            server.add_jobs(to_worker)

            if args['negative_sampling']:
                to_worker['firstrandom'] = True
                print("ADDING JOB")
                server.add_jobs(to_worker)
            time_sending += time.time() - _st

        # time_base_collection_and_sending = time.time() - _st

        print("WAITING FOR RESULTS")
        list_results, workers_logs  = server.get_job_results(
            current_update, 
            args['n_packages'],
            negative=False,
            prefix=RESULT_PREFIX,
            )
        time_total_rollouts = time.time() - _st
        
        original_experience = stack_experiences(exp_list)
        
        mc_oar = jax.tree_util.tree_map(
            lambda *dicts: jnp.concatenate(dicts, axis=0),*[x['mc_oar'] for x in list_results],)
        
        if args['negative_sampling']:
            list_negative_results, _  = server.get_job_results(
                current_update, 
                args['n_packages'],
                negative=True,
                prefix=RESULT_PREFIX,
                )
            negative_oar = jax.tree_util.tree_map(
            lambda *dicts: jnp.concatenate(dicts, axis=0),*[x['mc_oar'] for x in list_negative_results],)

        else: negative_oar = None
        
        print("GOT SOME STUFF FROM WORKERS")

        base_oar = process_base_rollout_output(state.apply_fn, state.params, original_experience, args['train_constants'])

        oar = dict(
            observations=jnp.concatenate((base_oar['observations'], mc_oar['observations']), axis=0),
            actions=jnp.concatenate((base_oar['actions'], mc_oar['actions']), axis=0),
            returns=jnp.concatenate((base_oar['returns'], mc_oar['returns']), axis=0),
            )
        not_sampled_observations = jnp.concatenate(not_selected_observations_list, axis=0)
        sampling_masks = jnp.stack(sampling_masks)
        no_sampling_masks = jnp.stack(no_sampling_masks)

        
        _st = time.time()
        if args['train_constants']['q_updates'] is not None:
            args['train_constants'] = args['train_constants'].copy({
                        'qf_update_batch_size':args['train_constants']['qf_update_batch_size'],
                        'q_train_len':len(q_train_oar['observations']),
                        })
            prngkey, _ = jax.random.split(prngkey)
            q_train_oar, q_test_oar = general_train_test_split(
                base_oar=base_oar,
                mc_oar=mc_oar,
                negative_oar=negative_oar,
                sampling_masks=sampling_masks,
                no_sampling_masks=no_sampling_masks,
                prngkey=prngkey,
                test_ratio=args['train_constants']['qf_test_ratio'],
                k=args['K'],
                nw=args['num_workers'],
                num_steps=args['num_steps'],
                num_envs=args['num_envs'],
                use_base_traj_for_q=args['use_base_traj_for_q'],
                split_type=args['split_type'],          
            )
            state, (q_loss, q_loss_dict) = q_step(
                state, 
                # trajectories, 
                q_train_oar, q_test_oar, # (Experience(original trajectory), List[dicts](kml trajs))
                prngkey,
                constant_params=args['train_constants'], jit_q_fn=jit_q_fn
                )
            state = state.replace(step=current_update)
        else:
            q_loss_dict = {}
        time_q_updates = time.time() - _st
        

        _st = time.time()
        prngkey, _ = jax.random.split(prngkey)
        p_train_data_dict = {
            'oar': oar,
            'base_oar': base_oar,
            'mc_oar': mc_oar,
            'not_sampled_observations': not_sampled_observations,
            }
        state, (loss, loss_dict) = p_step(
            state, 
            p_train_data_dict,
            prngkey,
            constant_params=args['train_constants'],
            )
        time_policy_updates = time.time() - _st

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
                "time/time_base_collection": time_base_collection,
                "time/time_sending": time_sending,
                "time/time_selecting": time_selecting,
                "time/time_total_rollouts" : time_total_rollouts,
                "time/time_policy_updates": time_policy_updates,
                "time/time_q_updates": time_q_updates,
                "time/time_evaluation": time_evaluation,
                'evaluation/score': eval_return,
                'evaluation/train_score': envs.get_last_return().mean()}, 
                commit=False, step=current_update)
            epoch_times = []

            loss_dict = jax.tree_map(lambda x: x.item(), loss_dict)
            q_loss_dict = jax.tree_map(lambda x: x.item(), q_loss_dict)
            loss_dict['loss'] = loss.item()
            wandb.log({'training/' + k: v for k, v in loss_dict.items()}, step=current_update)
            wandb.log({'q-training/' + k: v for k, v in q_loss_dict.items()}, step=current_update)
            wandb.log({'multihost/' + k: v for k, v in workers_logs.items()}, step=current_update)

        if args['verbose']:
            print({
                'time/timestep': timestep, 
                'time/updates': current_update, 
                'time/time': epoch_time,
                'time/num_interactions': interactions_per_epoch * current_update,
                "time/time_base_collection": time_base_collection,
                "time/time_sending": time_sending,
                "time/time_selecting": time_selecting,
                "time/time_total_rollouts" : time_total_rollouts,
                "time/time_policy_updates": time_policy_updates,
                "time/time_q_updates": time_q_updates,
                "time/time_evaluation": time_evaluation,
                'evaluation/score': eval_return,
                'evaluation/train_score': envs.get_last_return().mean()})
                
if __name__=='__main__':

    from args import args

    ctx = mp.get_context("forkserver")

    main(args)
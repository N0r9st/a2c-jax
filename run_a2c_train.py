import functools
import os
from copy import deepcopy

import jax
import numpy as np
import wandb

from jax_a2c.a2c import step
from jax_a2c.distributions import sample_action_from_normal as sample_action
from jax_a2c.env_utils import make_vec_env
from jax_a2c.evaluation import eval
from jax_a2c.policy import DiagGaussianPolicy
from jax_a2c.utils import (collect_experience, create_train_state,
                           process_experience)
from jax_a2c.saving import save_state, load_state


def main(args: dict):

    num_transition_steps = args['num_timesteps']//(args['num_envs'] * args['num_steps'])
    wandb_run_id = None
    start_update = 0

    envs = make_vec_env(
        name=args['env_name'], 
        num=args['num_envs'], 
        norm_r=args['norm_r'], 
        norm_obs=args['norm_obs'],)

    eval_envs = make_vec_env(
            name=args['env_name'], 
            num=args['num_envs'], 
            norm_r=False, 
            norm_obs=args['norm_obs'],)

    eval_envs.training=False

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
    
    @jax.jit
    def _policy_fn(prngkey, observation, params):
        values, (means, log_stds) = state.apply_fn({'params': params}, observation)
        sampled_actions  = sample_action(prngkey, means, log_stds)
        return values, sampled_actions
    
    eval_envs.training = False
    next_obs = envs.reset()
    next_obs_and_dones = (next_obs, np.array(next_obs.shape[0]*[False]))

    if args['load']:
        chkpnt = args['load']
        if os.path.exists(args['load']):
            print(f"Loading checkpoint {chkpnt}")
            state, additional = load_state(chkpnt, state)
            wandb_run_id = additional['wandb_run_id']
            start_update = state.step
        else:
            print(f"Checkpoint {chkpnt} not found!")

    if args['wb_flag']:
        if wandb_run_id is None:
            wandb.init(project=args['wandb_proj_name'], config=args)
            wandb_run_id = wandb.run.id
        else:
            wandb.init(project=args['wandb_proj_name'], config=args, id=wandb_run_id, resume="allow")

    total_updates = args['num_timesteps'] // ( args['num_envs'] * args['num_steps'])

    def get_timestep(current_update):
        return current_update * args['num_envs'] * args['num_steps']

    for current_update in range(start_update, total_updates):
        policy_fn = functools.partial(_policy_fn, params=state.params)
        if state.step%args['eval_every']==0:
            eval_envs.obs_rms = deepcopy(envs.obs_rms)
            _, eval_return = eval(state.apply_fn, state.params, eval_envs)
            if args['wb_flag']:
                wandb.log({'evaluation/score': eval_return}, commit=False,)
            print(f'Eval return: {eval_return}')

        prngkey, _ = jax.random.split(prngkey)
        next_obs_and_dones, experience = collect_experience(
            prngkey, 
            next_obs_and_dones, 
            envs, 
            num_steps=args['num_steps'], 
            policy_fn=policy_fn,)

        trajectories = process_experience(
            experience=experience,
            gamma=args['gamma'],
            lambda_=args['lambda_'])
            
        state, (loss, loss_dict) = step(
            state, 
            trajectories, 
            value_loss_coef=args['value_loss_coef'], 
            entropy_coef=args['entropy_coef'], 

            normalize_advantages=args['normalize_advantages'])

        if args['wb_flag'] and (current_update % args['log_freq']):
            wandb.log({'time/timestep': get_timestep(current_update), 'time/updates': current_update}, commit=False)

            loss_dict = jax.tree_map(lambda x: x.item(), loss_dict)
            loss_dict['loss'] = loss.item()
            wandb.log({'training/' + k: v for k, v in loss_dict.items()})

        if args['save'] and (current_update % args['save_every']):
            additional = {}
            additional['wandb_run_id'] = wandb_run_id
            save_state(args['save'], state, additional)

if __name__=='__main__':

    from args import args

    os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = args['allocate_memory']

    main(args)

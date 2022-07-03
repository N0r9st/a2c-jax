import functools
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import time
import jax
import multiprocessing as mp
import numpy as np

from jax_a2c.distributions import sample_action_from_normal as sample_action
from jax_a2c.env_utils import make_vec_env, DummySubprocVecEnv, run_workers_multihost as run_workers
from jax_a2c.policy import DiagGaussianPolicy, QFunction, DiagGaussianStateDependentPolicy, VFunction, DGPolicy
from jax_a2c.utils import create_train_state
from jax_a2c.km_mc_traj import km_mc_rollouts
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from multihost.job_server import KLMJobServer
from jax_a2c.utils import Experience

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

def collect_experience_withstate(
    prngkey,
    next_obs_and_dones_list: list,
    envs, 
    num_steps, 
    policy_fn, 
    ret_rms,
    obs_rms,
    initial_state_list: list,
    ):

    envs.training = True
    envs.ret_rms = ret_rms
    envs.obs_rms = obs_rms

    experience_list_out = []

    next_obs_and_dones_list_out = []
    initial_state_list_out = []
    for next_obs_and_dones, initial_state in zip(next_obs_and_dones_list, initial_state_list):
        envs.set_state(initial_state)
        next_observations, dones = next_obs_and_dones

        observations_list = []
        actions_list = []
        rewards_list = []
        values_list = []
        dones_list = [dones]
        states_list = [envs.get_state()]
        next_observations_list = []


        
        for _ in range(num_steps):
            observations = next_observations
            _, prngkey = jax.random.split(prngkey)
            values, actions = policy_fn(prngkey, observations) 
            actions = np.array(actions)
            next_observations, rewards, dones, info = envs.step(actions)
            observations_list.append(observations)
            actions_list.append(np.array(actions))
            rewards_list.append(rewards)
            values_list.append(values)
            dones_list.append(dones)
            states_list.append(envs.get_state())
            next_observations_list.append(next_observations)
            
            

        _, prngkey = jax.random.split(prngkey)
        values, actions = policy_fn(prngkey, next_observations) 
        values_list.append(values)

        experience = Experience(
            observations=np.stack(observations_list),
            actions=np.stack(actions_list),
            rewards=np.stack(rewards_list),
            values=np.stack(values_list),
            dones=np.stack(dones_list),
            states=states_list,
            next_observations=np.stack(next_observations_list)
        )
        experience_list_out.append(experience)
        next_obs_and_dones_list_out.append((next_observations, dones))
        initial_state_list_out.append(states_list[-1])
    starter_info = dict(
            initial_state_list=initial_state_list_out,
            next_obs_and_dones_list=next_obs_and_dones_list_out,
            prngkey=prngkey,
            obs_rms=envs.obs_rms,
            ret_rms=envs.ret_rms
        )
    return starter_info, experience_list_out

def _worker(remote, k_remotes, parent_remote, spaces, device, add_args) -> None:
    print('D:', device)
    server = KLMJobServer(host=add_args['redis_host'], port=add_args['redis_port'], password='fuckingpassword',
        base_prefix="manyenvs_")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    parent_remote.close()
    k_envs = DummySubprocVecEnv(remotes=k_remotes)
    
    k_envs.observation_space, k_envs.action_space = spaces
    k_envs = VecNormalize(k_envs, training=False)
    collect_experience_withstate_ = functools.partial(collect_experience_withstate, envs=k_envs)

    jit_value_and_policy_fn = jax.jit(add_args['policy_fn'], static_argnames=('determenistic',))

    while True:
        try:
            args = server.get_job()
            st = time.time()
            iteration = args.pop('iteration')
            print(f'GOT JOB FROM {iteration} ITERATION')

            policy_fn = functools.partial(
                jit_value_and_policy_fn, 
                params=args['policy_params'],
                vf_params=args['vf_params'])

            experiences = collect_experience_withstate_(policy_fn=policy_fn, **args['args'])
            
            result = dict(
                iteration=iteration,
                experiences=experiences,
            )
            server.commit_result(result, prefix=args['prefix'], negative=False)
            print(f'COMMITED RESULT FROM {iteration} ITERATION, TIME={time.time() - st:.1f}', sep="")            
        except EOFError:
            break

def main(args: dict):
    args['async'] = True
    if not args['split_between_devices']:
        os.environ['CUDA_VISIBLE_DEVICES'] = args['device']

    num_transition_steps = args['num_timesteps']//(args['num_envs'] * args['num_steps'])

    envs = make_vec_env(
        name=args['env_name'], 
        num=2, 
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
    
    # -----------------------------------------
    #            STARTING WORKERS
    #-----------------------------------------
    add_args = {
        'policy_fn': _apply_value_and_policy_fn,
        "redis_host": args["redis_host"], "redis_port": args["redis_port"]}
    run_workers(
        _worker, 
        None,
        k_envs_fn, 
        args['num_workers'], 
        (envs.observation_space, envs.action_space),
        ctx,
        split_between_devices=args['split_between_devices'],
        add_args=add_args)
    # ------------------------------------------

    while True:
        time.sleep(5)
    
if __name__=='__main__':

    from args import args
    ctx = mp.get_context("forkserver")
    main(args)

import os
from unicodedata import name

from torch import ge
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.10'
# from multiprocessing import Pool
from multiprocessing.pool import ThreadPool as Pool
import time
from jax_a2c.env_utils import make_vec_env


import functools
from copy import deepcopy
import time

import jax
import numpy as np
from jax_a2c.distributions import sample_action_from_normal as sample_action
from jax_a2c.env_utils import make_vec_env
from jax_a2c.policy import DiagGaussianPolicy
from jax_a2c.utils import (collect_experience, create_train_state,
                           process_experience)


from jax_a2c.utils import get_mc_returns, process_rewards
import jax
import jax.numpy as jnp


# policy_fn = None

# @functools.partial(jax.jit, static_argnums=(3,))

# from jax import api

@functools.partial(jax.jit, static_argnums=(3,))
def _policy_fn(prngkey, observation, params, apply_fn):
    values, (means, log_stds) = apply_fn({'params': params}, observation)
    sampled_actions  = sample_action(prngkey, means, log_stds)
    return values, sampled_actions



def km_mc_rollouts_trajectories(prngkey, k_envs, experience, policy_fn, gamma, K, M, max_steps=1000):
    observations = experience.observations
    actions = experience.actions
    rewards = experience.rewards
    values = experience.values
    dones = experience.dones
    states = experience.states

    num_envs = experience.observations.shape[1]


    rep_returns = []
    rep_actions = []

    orig_mc_returns = get_mc_returns(rewards, dones, values[-1], gamma)
    # rint(orig_mc_returns.shape)

    iterations = 0

    rep_observations = jnp.concatenate([observations for _ in range(K * M)], axis=1) # (n_steps, K * M * num_envs , ...)
    rep_dones = jnp.concatenate([dones for _ in range(K * M)], axis=1) # (n_steps, K * M * num_envs)
    
    for i, (next_ob, done, state) in enumerate(zip(rep_observations, rep_dones, states)):
        _, prngkey = jax.random.split(prngkey)
        # next_ob  -> (K * M * num_envs , ...)
        # done  -> K * M * num_envs
        state = state * K * M


        # COLLECTING ------------
        k_envs.set_state(state)
        rewards_list = []
        ds = [done]

        cumdones = jnp.zeros(shape=(next_ob.shape[0],))

        for l in range(max_steps):
            ob = next_ob
            _, prngkey = jax.random.split(prngkey)
            if l == 0:
                _, acts = policy_fn(prngkey, ob[:K*num_envs]) # acts: K * num_envs
                rep_actions.append(acts)
                acts = jnp.concatenate([acts for _ in range(M)], axis=0) # acts: M * K * num_envs
                
            else:
                _, acts = policy_fn(prngkey, ob) 
            next_ob, rews, d, info = k_envs.step(acts)
            rewards_list.append(rews)
            ds.append(d)
            iterations += 1
            cumdones += d
            if cumdones.all():
                break

        bootstrapped_values, _ = policy_fn(prngkey, ob) 
        ds = jnp.stack(ds)
        rewards_list = jnp.stack(rewards_list)

        kn_returns = process_rewards(ds, rewards_list, bootstrapped_values[..., 0], gamma)
        kn_returns = kn_returns.reshape(M, kn_returns.shape[0]//M).mean(axis=0)
        rep_returns.append(kn_returns)

    rep_values = jnp.concatenate([values for _ in range(K)], axis=1)
    rep_actions = jnp.stack(rep_actions)
    rep_returns = jnp.stack(rep_returns)
    rep_advantages = rep_returns - rep_values[:-1]
    rep_observations = jnp.concatenate([observations for _ in range(K)], axis=1)

    trajectories = (rep_observations, rep_actions, rep_returns, rep_advantages)
    trajectory_len = rep_observations.shape[0] * rep_observations.shape[1]
    trajectories = tuple(map(
        lambda x: jnp.reshape(x, (trajectory_len,) + x.shape[2:]), trajectories))

    old_trajectories = (observations, actions, orig_mc_returns, orig_mc_returns - values[:-1])
    old_trajectory_len = observations.shape[0] * observations.shape[1]
    old_trajectories = tuple(map(
        lambda x: jnp.reshape(x, (old_trajectory_len,) + x.shape[2:]), old_trajectories))

    trajectories = [jnp.concatenate([t1, t2], axis=0) for t1, t2 in zip(trajectories, old_trajectories)]

    return trajectories

def km_p_mc_rollouts_trajectories(prngkey, k_envs, experience, policy_fn, gamma, K, M, steps_parralel, max_steps=1000):
    observations = experience.observations
    actions = experience.actions
    rewards = experience.rewards
    values = experience.values
    dones = experience.dones
    states = experience.states

    num_steps, num_envs = experience.observations.shape[:2]
    assert k_envs.num_envs == K*M*steps_parralel*num_envs


    rep_returns = []
    rep_actions = []

    orig_mc_returns = get_mc_returns(rewards, dones, values[-1], gamma)
    # rint(orig_mc_returns.shape)

    iterations = 0

    rep_observations = jnp.concatenate([observations for _ in range(M * K)], axis=1) # (n_steps, K * M * num_envs , ...)
    rep_dones = jnp.concatenate([dones for _ in range(M * K)], axis=1) # (n_steps, K * M * num_envs)
    # rep_states = np.concatenate([states for _ in range(M * K)], axis=1, dtype=object) # (n_steps, K * M * num_envs)
    rep_states = []
    for step_state in states:
        rep_states.append(step_state*M*K)

    # for i, (rep_obs_batch, rep_dones_batch, rep_states_batch) in enumerate(zip(
    #     *(zip(rep_observations, rep_dones, rep_states),)*steps_parralel
    #     )):
    # slices = np.arange(start=0, stop=num_steps, )
    for slc in range(0, num_steps, steps_parralel):
        # rep_obs_batch -> (steps_parralel, K*M*num_envs, ...)
        rep_obs_batch = rep_observations[slc:slc + steps_parralel]
        rep_dones_batch = rep_dones[slc:slc + steps_parralel]
        rep_states_batch = rep_states[slc:slc + steps_parralel]
        next_ob = rep_obs_batch.reshape((steps_parralel * M * K * num_envs, -1))
        done = rep_dones_batch.reshape((steps_parralel * M * K * num_envs,))
        # state = rep_states_batch.reshape((steps_parralel * M * K * num_envs,))
        state = []
        for step_state in rep_states_batch:
            state += step_state

        k_envs.set_state(state)
        rewards_list = []
        ds = [done]

        cumdones = jnp.zeros(shape=(next_ob.shape[0],))

        for l in range(max_steps):
            ob = next_ob
            _, prngkey = jax.random.split(prngkey)
            if l == 0:
                _, acts = policy_fn(prngkey, ob) # non-repeated acts: steps_parralel * M * K * num_envs
                acts_per_step = acts.reshape(steps_parralel, M*K*num_envs, -1)
                acts_per_step = acts_per_step[:, :K*num_envs, :]
                rep_actions.append(acts_per_step.reshape(steps_parralel, K * num_envs, -1))
                acts_per_step = jnp.concatenate([acts_per_step for _ in range(M)], axis=1)
                acts = acts_per_step.reshape((steps_parralel * M * K * num_envs, -1))                
            else:
                _, acts = policy_fn(prngkey, ob) 
            next_ob, rews, d, info = k_envs.step(acts)
            rewards_list.append(rews)
            ds.append(d)
            iterations += 1
            cumdones += d
            if cumdones.all():
                break

        bootstrapped_values, _ = policy_fn(prngkey, ob) 
        ds = jnp.stack(ds)
        rewards_list = jnp.stack(rewards_list)

        kn_returns = process_rewards(ds, rewards_list, bootstrapped_values[..., 0], gamma)
        # kn_returns = kn_returns.reshape(M, kn_returns.shape[0]//M).mean(axis=0)
        kn_returns = kn_returns.reshape((steps_parralel, M, K * num_envs,))\
            .mean(axis=1)#.reshape((steps_parralel * K * num_envs,))
        rep_returns.append(kn_returns)

    rep_values = jnp.concatenate([values for _ in range(K)], axis=1)
    rep_actions = jnp.concatenate(rep_actions, axis=0)
    rep_returns = jnp.concatenate(rep_returns, axis=0)
    rep_advantages = rep_returns - rep_values[:-1]
    rep_observations = jnp.concatenate([observations for _ in range(K)], axis=1)

    trajectories = (rep_observations, rep_actions, rep_returns, rep_advantages)
    trajectory_len = rep_observations.shape[0] * rep_observations.shape[1]
    trajectories = tuple(map(
        lambda x: jnp.reshape(x, (trajectory_len,) + x.shape[2:]), trajectories))

    old_trajectories = (observations, actions, orig_mc_returns, orig_mc_returns - values[:-1])
    old_trajectory_len = observations.shape[0] * observations.shape[1]
    old_trajectories = tuple(map(
        lambda x: jnp.reshape(x, (old_trajectory_len,) + x.shape[2:]), old_trajectories))

    trajectories = [jnp.concatenate([t1, t2], axis=0) for t1, t2 in zip(trajectories, old_trajectories)]

    return trajectories

def default():
    print('making envs ...')
    k_envs = make_vec_env(
        name='Walker2d-v3', 
        num=48, 
        norm_r=True,
        norm_obs=True)

    envs = make_vec_env(
        name='Walker2d-v3', 
        num=4, 
        norm_r=True,
        norm_obs=True)


    model = DiagGaussianPolicy(
        hidden_sizes=[64,64], 
        action_dim=envs.action_space.shape[0],
        init_log_std=0)

    prngkey = jax.random.PRNGKey(0)

    state = create_train_state(
        prngkey,
        model,
        envs,
        learning_rate=.0002,
        decaying_lr=True,
        max_norm=.5,
        decay=.99,
        eps=1e-6,
        train_steps=2_000_000
    )
    next_obs = envs.reset()
    next_obs_and_dones = (next_obs, np.array(next_obs.shape[0]*[False]))

    @jax.jit
    def _policy_fn(prngkey, observation, params):
        values, (means, log_stds) = state.apply_fn({'params': params}, observation)
        sampled_actions  = sample_action(prngkey, means, log_stds)
        return values, sampled_actions
    policy_fn = functools.partial(_policy_fn, params=state.params, apply_fn=state.apply_fn)

    print('collecting')
    next_obs_and_dones, experience = collect_experience(
            prngkey, 
            next_obs_and_dones, 
            envs, 
            num_steps=32, 
            policy_fn=policy_fn,)
    print('running traj ...')
    [print(len(x)) for x in experience]
    st = time.time()
    trajectories = km_mc_rollouts_trajectories(
                prngkey=prngkey,
                experience=experience,
                gamma=.99,
                k_envs=k_envs,
                policy_fn=policy_fn,
                max_steps=5,
                K=3,
                M=4)
    print(time.time() - st)


def pvine():
    print('making envs ...')
    k_envs = make_vec_env(
        name='Walker2d-v3', 
        num=48*4, 
        norm_r=True,
        norm_obs=True)

    envs = make_vec_env(
        name='Walker2d-v3', 
        num=4, 
        norm_r=True,
        norm_obs=True)


    model = DiagGaussianPolicy(
        hidden_sizes=[64,64], 
        action_dim=envs.action_space.shape[0],
        init_log_std=0)

    prngkey = jax.random.PRNGKey(0)

    state = create_train_state(
        prngkey,
        model,
        envs,
        learning_rate=.0002,
        decaying_lr=True,
        max_norm=.5,
        decay=.99,
        eps=1e-6,
        train_steps=2_000_000
    )
    next_obs = envs.reset()
    next_obs_and_dones = (next_obs, np.array(next_obs.shape[0]*[False]))

    @jax.jit
    def _policy_fn(prngkey, observation, params):
        values, (means, log_stds) = state.apply_fn({'params': params}, observation)
        sampled_actions  = sample_action(prngkey, means, log_stds)
        return values, sampled_actions
    policy_fn = functools.partial(_policy_fn, params=state.params, apply_fn=state.apply_fn)

    print('collecting')
    next_obs_and_dones, experience = collect_experience(
            prngkey, 
            next_obs_and_dones, 
            envs, 
            num_steps=32, 
            policy_fn=policy_fn,)
    print('running traj ...')
    st = time.time()

    trajectories = km_p_mc_rollouts_trajectories(
                prngkey=prngkey,
                experience=experience,
                gamma=.99,
                k_envs=k_envs,
                policy_fn=policy_fn,
                max_steps=5,
                K=3,
                M=4,
                steps_parralel=4)
    print(time.time() - st)

def pool_default():

    print('making envs ...')
    # k_envs = make_vec_env(
    #     name='Walker2d-v3', 
    #     num=12, 
    #     norm_r=True,
    #     norm_obs=True)

    l_k_envs = [make_vec_env(
        name='Walker2d-v3', 
        num=12, 
        norm_r=True,
        norm_obs=True) for _ in range(8)]

    envs = make_vec_env(
        name='Walker2d-v3', 
        num=4, 
        norm_r=True,
        norm_obs=True)


    model = DiagGaussianPolicy(
        hidden_sizes=[64,64], 
        action_dim=envs.action_space.shape[0],
        init_log_std=0)

    prngkey = jax.random.PRNGKey(0)

    state = create_train_state(
        prngkey,
        model,
        envs,
        learning_rate=.0002,
        decaying_lr=True,
        max_norm=.5,
        decay=.99,
        eps=1e-6,
        train_steps=2_000_000
    )
    next_obs = envs.reset()
    next_obs_and_dones = (next_obs, np.array(next_obs.shape[0]*[False]))

    policy_fn = jax.jit(functools.partial(_policy_fn, params=state.params, apply_fn=state.apply_fn))

    print('collecting')

    print('running traj ...')
    st = time.time()
    exps = []
    for _ in range(8):
        prngkey, _ = jax.random.split(prngkey)
        next_obs_and_dones, experience = collect_experience(
                prngkey, 
                next_obs_and_dones, 
                envs, 
                num_steps=4, 
                policy_fn=policy_fn,)
        exps.append(experience)


    with Pool(4) as pool:

        map_args = zip([prngkey]*8, l_k_envs, exps, [policy_fn]*8, [.99]*8,  [3]*8, [1]*8, [5]*8)
        pool.starmap(km_mc_rollouts_trajectories, map_args)

        # from itertools import starmap
        # list(starmap(km_mc_rollouts_trajectories, map_args))
    print(time.time() - st)

def f(t):
    i, env = t
    print(f"Process {i} started!")
    for _ in range(100):
        env.step([env.action_space.sample() for _ in range(env.num_envs)])




from multiprocess import Pool
import multiprocess as mp


from multiprocessing import Pool
import multiprocessing as mp

def _worker(env_fn, inp_q, out_q):
    env = env_fn()
    while True:
        args = inp_q.get()
        # env.sim.set_state(args.state)
        out = env.step(args.action)
        out_q.put(out)

from collections import namedtuple
Sending = namedtuple('Sending', ('state', 'action'))


def test_pool():
    from jax_a2c.env_utils import get_env_fns
    n_workers = 12
    print('Init envs')
    env_fns = get_env_fns('Walker2d-v3', num=n_workers)
    env = env_fns[0]()
    pool = Pool(n_workers)
    m = mp.Manager()
    inp_q = m.Queue()
    out_q = m.Queue()
    out = pool.starmap_async(_worker,zip(env_fns, [inp_q]*n_workers, [out_q]*n_workers))
    action = env.action_space.sample()
    state = env.sim.get_state()
    print('Start steps')
    for _ in range(4):
        st = time.time()
        [inp_q.put(Sending(state=0, action=action)) for _ in range(32*4*4)]
        out = [out_q.get() for _ in range(32*4*4)]
        print(time.time() - st)
        inp_q.empty()
        out_q.empty()
    pool.terminate()

def test_vecenv():
    print('Init envs')
    envs = make_vec_env('Walker2d-v3', num=32*4*4)
    actions = [envs.action_space.sample() for _ in range(32*4*4)]
    print('Start steps')
    for _ in range(4):
        st = time.time()
        envs.step(actions)
        print(time.time() - st)



if __name__=="__main__":
    test_pool()
    # test_vecenv()
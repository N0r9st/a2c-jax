import functools
from jax_a2c.utils import get_mc_returns, process_rewards
import jax
import jax.numpy as jnp

@functools.partial(jax.jit, static_argnums=(1,))
def flatten_and_repeat(array, K):
    repeated = jnp.concatenate([array for _ in range(K)], axis=1) 
    return repeated.reshape((repeated.shape[1]*repeated.shape[0],) + repeated.shape[2:])

def km_mc_rollouts_trajectories(prngkey, k_envs, experience, policy_fn, gamma, K, M, max_steps=1000):
    observations = experience.observations
    actions = experience.actions
    rewards = experience.rewards
    values = experience.values
    dones = experience.dones
    states = experience.states

    num_envs = k_envs.num_envs


    rep_returns = []

    orig_mc_returns = get_mc_returns(rewards, dones, values[-1], gamma)
    iterations = 0
    flat_values = flatten_and_repeat(values[:-1], K)
    flat_observations = flatten_and_repeat(observations, K)
    flat_dones = flatten_and_repeat(dones, K)

    _, flat_actions = policy_fn(prngkey, flat_observations)

    m_flat_observations = jnp.concatenate([flat_observations for _ in range(M)], axis=0)
    m_flat_actions = jnp.concatenate([flat_actions for _ in range(M)], axis=0)
    m_flat_dones = jnp.concatenate([flat_dones for _ in range(M)], axis=0)


    flat_states = []
    for step_state in states:
        flat_states += step_state * K
    m_flat_states = flat_states * M

    for slc in range(0, len(m_flat_observations), num_envs):
        next_ob = m_flat_observations[slc: slc + num_envs]
        acts = m_flat_actions[slc: slc + num_envs]
        done = m_flat_dones[slc: slc + num_envs]
        state = m_flat_states[slc: slc + num_envs]
        k_envs.set_state(state)

        rewards_list = []
        ds = [done]

        cumdones = jnp.zeros(shape=(next_ob.shape[0],))

        for l in range(max_steps):
            ob = next_ob
            _, prngkey = jax.random.split(prngkey)
            if l != 0:                
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
        rep_returns.append(kn_returns)

    rep_returns = jnp.concatenate(rep_returns, axis=0)
    rep_returns = rep_returns.reshape(M, rep_returns.shape[0]//M).mean(axis=0)
    rep_advantages = rep_returns - flat_values

    trajectories = (flat_observations, flat_actions, rep_returns, rep_advantages)

    old_trajectories = (observations, actions, orig_mc_returns, orig_mc_returns - values[:-1])
    old_trajectory_len = observations.shape[0] * observations.shape[1]
    old_trajectories = tuple(map(
        lambda x: jnp.reshape(x, (old_trajectory_len,) + x.shape[2:]), old_trajectories))

    trajectories = [jnp.concatenate([t1, t2], axis=0) for t1, t2 in zip(trajectories, old_trajectories)]

    return trajectories

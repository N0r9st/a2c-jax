import functools
from jax_a2c.utils import process_rewards
import jax
import jax.numpy as jnp
import numpy as np
import os

@functools.partial(jax.jit, static_argnums=(1,))
def repeat(array, K):
    repeated = jnp.concatenate([array for _ in range(K)], axis=0) 
    return repeated# .reshape((repeated.shape[1]*repeated.shape[0],) + repeated.shape[2:])

def km_mc_rollouts(prngkey, k_envs, experience, policy_fn, gamma, K, M, max_steps=1000, firstrandom=False):
    observations = experience.observations # (num_steps, num_envs, obs_shape)
    dones = experience.dones
    states = experience.states

    num_envs = k_envs.num_envs

    iterations = 0
    flat_observations = repeat(observations, K)
    # (in_states, obs_shape) -> (K * in_states, obs_shape)
    flat_dones = repeat(dones, K)

    if firstrandom:
        flat_actions = jax.random.uniform(prngkey, minval=k_envs.action_space.low, maxval=k_envs.action_space.high, 
        shape=flat_observations.shape[:-1] + k_envs.action_space.shape)
    else:
        _, flat_actions = policy_fn(prngkey, flat_observations)

    m_flat_observations = jnp.concatenate([flat_observations for _ in range(M)], axis=0)
    # (K * in_states, obs_shape) -> (M * K * in_states, obs_shape)
    m_flat_actions = jnp.concatenate([flat_actions for _ in range(M)], axis=0)
    m_flat_dones = jnp.concatenate([flat_dones for _ in range(M)], axis=0)
    m_flat_states = (states * K) * M

    all_rewards_array = np.zeros((max_steps, m_flat_observations.shape[0]))
    all_ds_array = np.zeros((max_steps + 1,) + m_flat_dones.shape)
    all_obs_array = np.zeros((max_steps,) + m_flat_observations.shape)
    all_next_obs_array = np.zeros((max_steps,) + m_flat_observations.shape)
    all_act_array = np.zeros((max_steps,) + m_flat_actions.shape)
    all_boot_list = []

    all_ds_array[0] = m_flat_dones

    for slc in range(0, len(m_flat_observations), num_envs):
        next_ob = m_flat_observations[slc: slc + num_envs]
        acts = m_flat_actions[slc: slc + num_envs]
        state = m_flat_states[slc: slc + num_envs]
        k_envs.set_state(state)

        cumdones = jnp.zeros(shape=(next_ob.shape[0],))

        for l in range(max_steps):
            ob = next_ob
            _, prngkey = jax.random.split(prngkey)
            if l != 0:                
                _, acts = policy_fn(prngkey, ob) 
            
            acts = np.array(acts)
            for _ in range(1 + (os.environ.get('DOUBLE_STEP') is not None)):
                next_ob, rews, d, info = k_envs.step(acts)
            all_rewards_array[l, slc: slc + num_envs] = rews
            all_ds_array[l + 1, slc: slc + num_envs] = d
            all_obs_array[l, slc: slc + num_envs] = ob
            all_next_obs_array[l, slc: slc + num_envs] = next_ob
            all_act_array[l, slc: slc + num_envs] = acts
            iterations += 1
            cumdones += d
            if cumdones.all():
                break
        
        bootstrapped_values, _ = policy_fn(prngkey, next_ob) # ???
        all_boot_list.append(bootstrapped_values)
    
    rollout_data = dict(
        observations=all_obs_array,
        next_observations=all_next_obs_array,
        actions=all_act_array,
        dones=all_ds_array,
        rewards=all_rewards_array,
        bootstrapped=jnp.concatenate(all_boot_list)[..., 0], # (1, in_states)
        )
    return rollout_data

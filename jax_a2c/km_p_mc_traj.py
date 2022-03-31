from jax_a2c.utils import get_mc_returns, process_rewards
import jax
import jax.numpy as jnp
import numpy as np

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

from jax_a2c.utils import get_mc_returns, process_rewards
import jax
import jax.numpy as jnp

def k_mc_rollouts_trajectories(prngkey, k_envs, experience, policy_fn, gamma, max_steps=1000):
    observations = experience.observations
    actions = experience.actions
    rewards = experience.rewards
    values = experience.values
    dones = experience.dones
    states = experience.states


    rep_returns = []
    rep_actions = []

    K = k_envs.num_envs // observations.shape[1]

    orig_mc_returns = get_mc_returns(rewards, dones, values[-1], gamma)
    # rint(orig_mc_returns.shape)

    iterations = 0

    rep_observations = jnp.concatenate([observations for _ in range(K)], axis=1)
    rep_dones = jnp.concatenate([dones for _ in range(K)], axis=1)
    
    for i, (next_ob, done, state) in enumerate(zip(rep_observations, rep_dones, states)):
        _, prngkey = jax.random.split(prngkey)
        # next_ob = jnp.concatenate([obs for _ in range(K)])
        # done = jnp.concatenate([done for _ in range(K)])
        state = state * K


        # COLLECTING ------------
        k_envs.set_state(state)
        actions_list = []
        rewards_list = []
        values_list = []
        ds = [done]

        cumdones = jnp.zeros(shape=(next_ob.shape[0],))

        for _ in range(max_steps):
            ob = next_ob
            _, prngkey = jax.random.split(prngkey)
            _, acts = policy_fn(prngkey, ob) 
            next_ob, rews, d, info = k_envs.step(acts)
            actions_list.append(acts)
            rewards_list.append(rews)
            ds.append(d)
            iterations += 1
            cumdones += d
            if cumdones.all():
                break

        bootstrapped_values, _ = policy_fn(prngkey, ob) 
        values_list.append(bootstrapped_values[..., 0])
        ds = jnp.stack(ds)
        rewards_list = jnp.stack(rewards_list)

        kn_returns = process_rewards(ds, rewards_list, bootstrapped_values[..., 0], gamma)
        rep_returns.append(kn_returns)
        rep_actions.append(actions_list[0])

    rep_values = jnp.concatenate([values for _ in range(K)], axis=1)
    rep_actions = jnp.stack(rep_actions)
    rep_returns = jnp.stack(rep_returns)
    rep_advantages = rep_returns - rep_values[:-1]


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

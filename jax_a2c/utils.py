import functools
from typing import Any, Callable, Dict, Tuple
from collections import namedtuple
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

import jax_a2c.env_utils

Array = Any
PRNGKey = Any
ModelClass = Any
Experience = namedtuple(
    'Experience', 
    ['observations', 'actions', 'rewards', 'values', 'dones', 'states', 'next_observations'])

class QTrainState(TrainState):
    q_fn: Callable = struct.field(pytree_node=False)
    

def create_train_state(
    prngkey: PRNGKey, 
    policy_model: ModelClass,
    qf_model: ModelClass,
    envs: jax_a2c.env_utils.SubprocVecEnv,
    learning_rate: float, 
    decaying_lr: bool, 
    max_norm: float,
    decay: float,
    eps: float,
    train_steps: int = 0) -> QTrainState:

    dummy_input = envs.reset()
    dummy_action = np.stack([envs.action_space.sample() for _ in range(envs.num_envs)])

    policy_variables = policy_model.init(prngkey, dummy_input)
    policy_params = policy_variables['params']

    prngkey, _ = jax.random.split(prngkey)
    qf_variables = qf_model.init(prngkey, dummy_input, dummy_action)
    qf_params = qf_variables['params']

    if decaying_lr:
        lr = optax.linear_schedule(
            init_value = learning_rate, end_value=0.,
            transition_steps=train_steps)
    else:
        lr = learning_rate

    tx = optax.chain(
        optax.clip_by_global_norm(max_norm),
        optax.rmsprop(learning_rate=lr, decay=decay, eps=eps)
        )
    state = QTrainState.create(
        apply_fn=policy_model.apply,
        params={'policy_params': policy_params, 'qf_params': qf_params},
        q_fn=qf_model.apply,
        tx=tx)

    return state

@jax.jit
@functools.partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def gae_advantages(
        rewards: np.ndarray,
        terminal_masks: np.ndarray,
        values: np.ndarray,
        discount: float,
        gae_param: float):
    assert rewards.shape[0] + 1 == values.shape[0]
    advantages = []
    gae = 0.
    for t in reversed(range(len(rewards))):
        value_diff = discount * values[t + 1] * terminal_masks[t + 1] - values[t]
        delta = rewards[t] + value_diff
        gae = delta + discount * gae_param * terminal_masks[t + 1] * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    return jnp.array(advantages)

def collect_experience(
    prngkey: PRNGKey,
    next_obs_and_dones: Array,
    envs: jax_a2c.env_utils.SubprocVecEnv, 
    num_steps: int, 
    policy_fn: Callable, 
    )-> Tuple[Array, ...]:

    envs.training = True

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
    return (next_observations, dones), experience

@functools.partial(jax.jit, static_argnums=(1,2))
def process_experience(
    experience: Tuple[Array, ...], 
    gamma: float = .99, 
    lambda_: float = .95,
    ):
    observations = experience.observations
    actions = experience.actions
    rewards = experience.rewards
    values = experience.values
    dones = experience.dones

    dones = jnp.logical_not(dones).astype(float)
    advantages = gae_advantages(rewards, dones, values, gamma, lambda_)
    returns = advantages + values[:-1]
    trajectories = (observations, actions, returns, advantages)
    num_agents, actor_steps = observations.shape[:2]
    trajectory_len = num_agents * actor_steps
    trajectories = tuple(map(
        lambda x: np.reshape(x, (trajectory_len,) + x.shape[2:]), trajectories))
    return trajectories

@functools.partial(jax.jit, static_argnums=(1,3,4,5,6))
def process_experience_with_entropy(
    experience: Tuple[Array, ...], 
    apply_fn,
    params,
    gamma: float = .99, 
    lambda_: float = .95,
    alpha: float = .1,
    entropy: str = 'estimation', # estimation/real
    ):
    # observations, actions, rewards, values, dones = experience
    observations = experience.observations
    actions = experience.actions
    rewards = experience.rewards
    values = experience.values
    dones = experience.dones
    next_observations = experience.next_observations

    # ------- LOGPROBS ----------
    actor_steps, num_agents = observations.shape[:2]

    
    if alpha>0:
        _, (means, log_stds) = apply_fn(
            {'params': params}, 
            observations.reshape((actor_steps* num_agents,) + observations.shape[2:]))
        if entropy == 'estimation':
            logprobs = calculate_action_logprobs(actions.reshape((actor_steps * num_agents, -1)), means, log_stds)
            entropy = - logprobs.reshape((actor_steps, num_agents))
        elif entropy == 'real':
            entropy = (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_stds).sum(-1).reshape((actor_steps, num_agents))
    else:
        entropy = 0
    entropy_rewards = rewards + alpha * entropy
    #------------------------------

    dones = jnp.logical_not(dones).astype(float)
    advantages = gae_advantages(entropy_rewards, dones, values, gamma, lambda_)
    returns = advantages + values[:-1]
    trajectories = (observations, actions, returns, advantages, next_observations, dones[1:], rewards)
    trajectory_len = num_agents * actor_steps
    trajectories = tuple(map(
        lambda x: jnp.reshape(x, (trajectory_len,) + x.shape[2:]), 
        trajectories
        ))
    return trajectories, entropy

@functools.partial(jax.jit, static_argnames=('gamma',))
def process_rewards(dones, rewards, bootstrapped_values, gamma):
    masks = jnp.cumprod((1-dones)*gamma, axis=0)/gamma
    k_returns = (rewards*masks[:-1]).sum(axis=0) + bootstrapped_values * masks[-1]
    return k_returns

# @functools.partial(jax.jit, static_argnums=(3, 4))
def process_rewards_with_entropy(
    apply_fn, 
    params, 
    observations, 
    actions, 
    dones, 
    rewards, 
    bootstrapped_values, 
    alpha, 
    gamma, 
    entropy: str, # estimation / real
    ):
    """ Should be used inside loss_fn
    """

    masks = jnp.cumprod((1-dones)*gamma, axis=0)/gamma
    
    if alpha>0:
        len_rollout, n_rollout, obs_shape = observations.shape
        _, (means, log_stds) = apply_fn({'params': params}, observations.reshape((len_rollout * n_rollout, obs_shape)))
        if entropy == 'estimation':
            actions = actions.reshape((len_rollout * n_rollout, -1))
            logprobs = calculate_action_logprobs(actions, means, log_stds)
            entropy = - logprobs.reshape((len_rollout, n_rollout))
        elif entropy == 'real':
            entropy = (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_stds).sum(-1).reshape((len_rollout, n_rollout,))
    else:
        entropy = 0

    rewards = rewards + alpha * entropy
    k_returns = (rewards * masks[:-1]).sum(axis=0) + bootstrapped_values * masks[-1]
    return k_returns

vmap_process_rewards_with_entropy = jax.vmap(
    process_rewards_with_entropy, in_axes=(None, None, 0, 0, 0, 0, 0, None, None, None), out_axes=0,
    )

def calculate_action_logprobs(actions, means, log_stds):
    stds = jnp.exp(log_stds)
    pre_tanh_logprobs = -(actions-means)**2/(2*stds**2) - jnp.log(2*jnp.pi)/2 - log_stds
    action_logprobs = (pre_tanh_logprobs).sum(axis=-1)
    return action_logprobs

@functools.partial(jax.jit, static_argnums=(3,))
def get_mc_returns(rewards, dones, last_values, gamma):
    masks = 1-dones
    returns = jnp.zeros_like(rewards)
    returns = returns.at[-1].set(last_values)
    for i, rews in reversed(list(enumerate(rewards))):
        returns = returns.at[i].set(rews + returns[i+1]*gamma*masks[i+1])
    return returns

@jax.jit
def concat_trajectories(traj_list):
    return [jnp.concatenate(x, axis=0) for x in zip(*traj_list)]

@functools.partial(jax.jit)
def stack_experiences(exp_list):
    num_steps = exp_list[0].observations.shape[0]
    last_vals = exp_list[-1][3][-1]
    last_dones = exp_list[-1][4][-1]
    last_states = exp_list[-1][5][-1]
    observations = jnp.concatenate([x.observations for x in exp_list], axis=0)
    actions = jnp.concatenate([x.actions for x in exp_list], axis=0)
    rewards = jnp.concatenate([x.rewards for x in exp_list], axis=0)
    values = jnp.concatenate([x.values[:num_steps] for x in exp_list], axis=0)
    dones = jnp.concatenate([x.dones[:num_steps] for x in exp_list], axis=0)
    next_observations = jnp.concatenate([x.next_observations for x in exp_list], axis=0)
    states = []
    for st in exp_list:
        states += st.states[:num_steps]
    values = jnp.append(values, last_vals[None], axis=0)
    dones = jnp.append(dones, last_dones[None], axis=0)
    states.append(last_states)
    return Experience(
        observations=observations,
        actions=actions,
        rewards=rewards,
        values=values,
        dones=dones,
        states=states,
        next_observations=next_observations
    )


@jax.jit
def flatten_experience(experience: Experience): 
    num_steps, num_envs = experience.observations.shape[:2]
    return Experience(
        observations = experience.observations[:num_steps].reshape((num_envs*num_steps,) + experience.observations.shape[2:]),
        actions = experience.actions[:num_steps].reshape((num_envs*num_steps,) + experience.actions.shape[2:]),
        rewards = experience.rewards[:num_steps].reshape((num_envs*num_steps,) + experience.rewards.shape[2:]),
        values = experience.values[:num_steps].reshape((num_envs*num_steps,) + experience.values.shape[2:]),
        dones = experience.dones[:num_steps].reshape((num_envs*num_steps,) + experience.dones.shape[2:]),
        states = flatten_list(experience.states[:num_steps]),
        next_observations=experience.next_observations.reshape((num_envs*num_steps,) + experience.next_observations.shape[2:])
            )

def select_random_states(prngkey, n, experience, type, **kwargs):
    flattened = flatten_experience(experience)
    if type=='uniform':
        p = None
    if type=='adv':
        advs = kwargs['advantages'].reshape(-1)
        p = jax.nn.softmax((advs**2)/kwargs['sampling_prob_temp'], axis=0)
    return select_experience_random(prngkey, n, flattened, p=p)

# @functools.partial(jax.jit, static_argnames=('replace', 'n'))
def select_experience_random(prngkey, n, experience, replace=False, p=None): 
    num_states = len(experience[0])
    choices = jnp.array(jax.random.choice(prngkey, num_states, shape=(n,), replace=replace, p=p))
    not_selected_indices = jnp.delete(jnp.arange(0, num_states), choices)
    selected = Experience(observations=experience.observations[choices],
                    actions=experience.actions[choices],
                    rewards=experience.rewards[choices],
                    values=experience.values[choices],
                    dones=experience.dones[choices],
                    states=substract_from_list(experience.states, choices),
                    next_observations=experience.next_observations[choices]
                        )
    return selected, experience.observations[not_selected_indices], choices, not_selected_indices

def flatten_list(lst):
    out = []
    for l in lst:
        out += l
    return out
def substract_from_list(lst, ind):
    out = []
    for i in ind:
        out.append(lst[i])
    return out

@functools.partial(jax.jit, static_argnames=('M',))
def process_mc_rollouts(observations, actions, returns, M):
    returns = returns.reshape(M, returns.shape[0]//M).mean(axis=0)
    observations = observations[0, :observations.shape[1]//M]
    actions = actions[0, :actions.shape[1]//M]
    return observations, actions, returns

vmap_process_mc_rollouts = jax.vmap(
    process_mc_rollouts, in_axes=(0, 0, 0, None), out_axes=0,
    )

@functools.partial(jax.jit, static_argnames=('constant_params','apply_fn'))
def process_rollout_output(apply_fn, params, data_tuple, constant_params):
    orig_exp, mc_rollouts_exp = data_tuple
    (observations, 
    actions, returns_loggrad, _, 
    next_observations, next_dones, rewards), entropy = process_experience_with_entropy(
        orig_exp, 
        apply_fn,
        params['policy_params'],
        lambda_=constant_params['lambda_'], 
        gamma=constant_params['gamma'],
        alpha=constant_params['alpha'],
        entropy=constant_params['entropy'],
        )
    if constant_params['type'] != 'standart':
        mc_rollouts_returns = vmap_process_rewards_with_entropy(
            apply_fn,
            params['policy_params'],
            mc_rollouts_exp['observations'],
            mc_rollouts_exp['actions'],
            mc_rollouts_exp['dones'],
            mc_rollouts_exp['rewards'],
            mc_rollouts_exp['bootstrapped'],
            constant_params['alpha'],
            constant_params['gamma'],
            constant_params['entropy'],
        )

        mc_observations, mc_actions, mc_returns = vmap_process_mc_rollouts(
            mc_rollouts_exp['observations'],
            mc_rollouts_exp['actions'],
            mc_rollouts_returns,
            constant_params['M']
        )
        mc_observations, mc_actions, mc_returns = tuple(map(
            lambda x: x.reshape((x.shape[0]*x.shape[1],) + x.shape[2:]), (mc_observations, mc_actions, mc_returns)
        ))

        observations = jnp.concatenate((observations, mc_observations), axis=0)
        actions = jnp.concatenate((actions, mc_actions), axis=0)
        returns_loggrad = jnp.concatenate((returns_loggrad, mc_returns), axis=0)
    return dict(
        observations=observations,
        actions=actions,
        returns=returns_loggrad,
    )
@functools.partial(jax.jit, static_argnames=('constant_params','apply_fn'))
def process_mc_rollout_output(apply_fn, params, mc_rollouts_exp, constant_params):
    mc_rollouts_returns = vmap_process_rewards_with_entropy(
        apply_fn,
        params['policy_params'],
        mc_rollouts_exp['observations'],
        mc_rollouts_exp['actions'],
        mc_rollouts_exp['dones'],
        mc_rollouts_exp['rewards'],
        mc_rollouts_exp['bootstrapped'],
        constant_params['alpha'],
        constant_params['gamma'],
        constant_params['entropy'],
    )

    mc_observations, mc_actions, mc_returns = vmap_process_mc_rollouts(
        mc_rollouts_exp['observations'],
        mc_rollouts_exp['actions'],
        mc_rollouts_returns,
        constant_params['M']
    )
    mc_observations, mc_actions, mc_returns = tuple(map(
        lambda x: x.reshape((x.shape[0]*x.shape[1],) + x.shape[2:]), (mc_observations, mc_actions, mc_returns)
    ))
    return dict(
        observations=mc_observations,
        actions=mc_actions,
        returns=mc_returns,
    )

@functools.partial(jax.jit, static_argnames=('constant_params','apply_fn'))
def process_single_mc_rollout_output(mc_rollouts_exp, constant_params):
    mc_rollouts_returns = process_rewards_with_entropy(
        None,
        None,
        mc_rollouts_exp['observations'],
        mc_rollouts_exp['actions'],
        mc_rollouts_exp['dones'],
        mc_rollouts_exp['rewards'],
        mc_rollouts_exp['bootstrapped'],
        constant_params['alpha'],
        constant_params['gamma'],
        constant_params['entropy'],
    )

    mc_observations, mc_actions, mc_returns = process_mc_rollouts(
        mc_rollouts_exp['observations'],
        mc_rollouts_exp['actions'],
        mc_rollouts_returns,
        constant_params['M']
    )
    # mc_observations, mc_actions, mc_returns = tuple(map(
    #     lambda x: x.reshape((x.shape[0]*x.shape[1],) + x.shape[2:]), (mc_observations, mc_actions, mc_returns)
    # ))
    return dict(
        observations=mc_observations,
        actions=mc_actions,
        returns=mc_returns,
    )

@functools.partial(jax.jit, static_argnames=('constant_params','apply_fn'))
def process_base_rollout_output(apply_fn, params, orig_exp, constant_params):
    (observations, 
    actions, returns_loggrad, _, 
    next_observations, next_dones, rewards), entropy = process_experience_with_entropy(
        orig_exp, 
        apply_fn,
        params['policy_params'],
        lambda_=constant_params['lambda_'], 
        gamma=constant_params['gamma'],
        alpha=constant_params['alpha'],
        entropy=constant_params['entropy'],
        )
    return dict(
        observations=observations,
        actions=actions,
        returns=returns_loggrad,
    )

def calculate_interactions_per_epoch(args):
    num_interactions = args['num_envs'] * args['num_envs']
    if args['type'] == 'standart':
        return num_interactions
    else:
        num_interactions += args['n_samples'] * args['K'] * args['M'] * args['L']
        if args['negative_sampling']:
            num_interactions += args['n_samples'] * args['K'] * args['M'] * args['L']

    return num_interactions
import functools
import jax 
import jax.numpy as jnp

@functools.partial(jax.jit, static_argnums=(1,))
def evaluate_actions_norm_with_repeats(params, apply_fn, observations, actions):
    """
    actions.shape = 
    """
    values, (means, log_stds) = apply_fn({'params': params}, observations)
    means = means[:, None, :]
    log_stds = log_stds[:, None, :]
    stds = jnp.exp(log_stds)
    pre_tanh_logprobs = -(actions-means)**2/(2*stds**2) - jnp.log(2*jnp.pi)/2 - log_stds
    action_logprobs = (pre_tanh_logprobs).sum(axis=-1)
    dist_entropy = (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_stds).sum(-1).mean()
    return action_logprobs, values[..., 0], dist_entropy, log_stds  

def sample_action_from_normal(prngkey, normal_means, log_normal_stds):
    """ Normal dist parameters -> samples and logprobs for Normal
    """
    normal_stds = jnp.exp(log_normal_stds)
    normal_samples = normal_means + normal_stds * jax.random.normal(prngkey, shape=normal_means.shape)
    return normal_samples

def evaluate_actions_norm(params, apply_fn, observations, actions, prngkey):
    values, (means, log_stds) = apply_fn({'params': params}, observations)
    stds = jnp.exp(log_stds)
    pre_tanh_logprobs = -(actions-means)**2/(2*stds**2) - jnp.log(2*jnp.pi)/2 - log_stds
    action_logprobs = (pre_tanh_logprobs).sum(axis=-1)
    dist_entropy = (0.5 + 0.5 * jnp.log(2 * jnp.pi) + log_stds).sum(-1).mean()
    # --------
    action_samples = sample_action_from_normal(prngkey, means, log_stds)
    pre_tanh_logprobs = -(action_samples-means)**2/(2*stds**2) - jnp.log(2*jnp.pi)/2 - log_stds
    sampled_action_logprobs = (pre_tanh_logprobs).sum(axis=-1)
    # --------
    return action_logprobs, sampled_action_logprobs, values[..., 0], dist_entropy, log_stds, action_samples  
    
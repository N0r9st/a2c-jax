# Advantage Actor Critic (A2C): jax + flax implementation

Current version supports only environments with continious action spaces and was tested on mujoco 1.50 environments.  
Algorighn uses wandb logging.   

A2C uses a diagonal gaussian policy with state-independent action distribution variance.

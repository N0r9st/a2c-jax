import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--environment', type=str, default='HalfCheetah-v3')
    args = parser.parse_args()
    return args


args = dict(
        seed=777,
        gamma=.99,
        lambda_=1., # gae lambda coef
        lr=2e-3,
        linear_decay=True,
        value_loss_coef=.4,
        entropy_coef=0.,
        eval_every=10,
        wb_flag=True, # log to wandb or not
        hidden_sizes=(64, 64,),
        env_name='HalfCheetah-v3',
        num_envs=4,
        num_steps=32,
        num_timesteps=2_000_000,
        max_grad_norm=.5,
        rms_beta2=.99,
        rms_eps=1e-5,
        init_log_std=0., 
        norm_r=True,
        norm_obs=True,
        normalize_advantages=False,
        prefix='', # prefix for wandb name
        device='0',
        allocate_memory='.15', 
        wandb_proj_name='test_jax_a2c',
        log_freq=5,
    )

cmd_args = parse_args()

def update(args, cmd_args):
    args['wandb_proj_name'] = cmd_args.wandb_project
    
    if cmd_args.wandb_project:
        args['wb_flag'] = True
    else:
        args['wb_flag'] = False

    args['env_name'] = cmd_args.environment
    return args

args = update(args, cmd_args)
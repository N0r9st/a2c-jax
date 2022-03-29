import argparse
possible_types = ['standart', 'K-rollouts', 'KM-rollouts']
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-project', type=str, default=None)
    parser.add_argument('--environment', type=str, default='HalfCheetah-v3')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=100)

    parser.add_argument('--type', type=str, default='standart',)

    parser.add_argument('--K', type=int, default=0)
    parser.add_argument('--L', type=int, default=0)
    parser.add_argument('--M', type=int, default=0)
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
        eval_every=50,
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
        device='0',
        allocate_memory='.15', 
        wandb_proj_name='test_jax_a2c',
        log_freq=50,
    )

cmd_args = parse_args()

def update(args, cmd_args):
    args['wandb_proj_name'] = cmd_args.wandb_project
    
    if cmd_args.wandb_project:
        args['wb_flag'] = True
    else:
        args['wb_flag'] = False

    args['env_name'] = cmd_args.environment
    args['device'] = cmd_args.device
    args['seed'] = cmd_args.seed

    args['load'] = cmd_args.load
    args['save'] = cmd_args.save
    args['save_every'] = cmd_args.save_every

    assert cmd_args.type in possible_types
    args['type'] = cmd_args.type

    args['K'] = cmd_args.K
    args['L'] = cmd_args.L
    args['M'] = cmd_args.L

    if args['type']=='standart':
        args['K'] = 0
        args['L'] = 0
        args['M'] = 0

    return args

args = update(args, cmd_args)
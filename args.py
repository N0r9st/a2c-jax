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

    parser.add_argument('--num-timesteps', type=int, default=2_000_000)

    parser.add_argument('--type', type=str, default='standart',)

    parser.add_argument('--K', type=int, default=0)
    parser.add_argument('--L', type=int, default=0)
    parser.add_argument('--M', type=int, default=0)


    parser.add_argument('--layers', type=str, default='64-64')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--linear-decay', action='store_true', default=False)
    parser.add_argument('--num-steps', type=int, default=32)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--value-loss-coef', type=float, default=0.4)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--num-k-envs', type=int, default=None)

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
        eval_every=20,
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
        log_freq=20,
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
    args['M'] = cmd_args.M

    if args['type']=='standart':
        args['K'] = 0
        args['L'] = 0
        args['M'] = 0

    if args['type']=='K-rollouts':
        args['M'] = 1

    args['num_timesteps'] = cmd_args.num_timesteps
    args['hidden_sizes'] = tuple(int(x) for x in cmd_args.layers.split('-'))
    args['lr'] = cmd_args.lr
    args['linear_decay'] = cmd_args.linear_decay
    args['num_steps'] = cmd_args.num_steps
    args['num_envs'] = cmd_args.num_envs
    args['value_loss_coef'] = cmd_args.value_loss_coef
    args['gamma'] = cmd_args.gamma

    args['log_freq'] = cmd_args.log_every
    args['eval_every'] = cmd_args.log_every

    if cmd_args.num_k_envs is None:
        args['num_k_envs'] = args['num_envs'] * args['K'] * args['M']
    else:
        args['num_k_envs'] = cmd_args.num_k_envs

    return args

args = update(args, cmd_args)

if __name__=='__main__':
    print(args)
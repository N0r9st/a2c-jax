import argparse
possible_types = ['sample-KM-rollouts-fast', 'standart']
possible_q_updates = [None, 'rep', 'log', 'none']
possible_policy_types = ['DiagGaussianPolicy', 'DiagGaussianStateDependentPolicy']
possible_sampling_types = ['uniform', 'adv',]
possible_split_types = ['full_tt_split',
                        'new_full_tt_split',
                        'weighted_tt_split',
                        'standart',]

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

    parser.add_argument('--type', type=str, default='sample-KM-rollouts-fast',)

    parser.add_argument('--K', type=int, default=0)
    parser.add_argument('--L', type=int, default=0)
    parser.add_argument('--M', type=int, default=0)


    parser.add_argument('--layers', type=str, default='64-64')
    parser.add_argument('--q-layers', type=str, default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--linear-decay', action='store_true', default=False)
    parser.add_argument('--num-steps', type=int, default=32)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--value-loss-coef', type=float, default=0.4)
    parser.add_argument('--q-loss-coef', type=float, default=0.4)
    parser.add_argument('--q-loss-multiplier', type=float, default=1.)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--num-k-envs', type=int, default=None)

    
    parser.add_argument('--num-updates', type=int, default=None)

    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--split-between-devices', action='store_true', default=False)


    parser.add_argument('--q-updates', type=str, default=None)
    parser.add_argument('--eval-with-q', action='store_true', default=False)

    parser.add_argument('--init-log-std', type=float, default=0.)
    parser.add_argument('--alpha', type=float, default=0.)
    parser.add_argument('--policy-type', type=str, default='DiagGaussianPolicy')

    parser.add_argument('--sampling-type', type=str, default='uniform')
    parser.add_argument('--sampling-prob-temp', type=float, default=1)
    parser.add_argument('--n-samples', type=int, default=0)
    parser.add_argument('--ignore-original-trajectory', action='store_true', default=False)
    parser.add_argument('--km-determenistic', action='store_true', default=False)

    parser.add_argument('--gradstop', type=str, default='full')

    parser.add_argument('--entropy', type=str, default='estimation')

    parser.add_argument('--q-targets', type=str, default='mc')
    parser.add_argument('--entropy-coef', type=float, default=0)
    parser.add_argument('--qf-update-batch-size', type=int, default=-1)
    parser.add_argument('--qf-update-epochs', type=int, default=1)
    parser.add_argument('--qf-test-ratio', type=float, default=.1)
    parser.add_argument('--use-samples-for-log-update', action='store_true', default=False)
    parser.add_argument('--negative-sampling', action='store_true', default=False)


    parser.add_argument('--use-q-tolerance', action='store_true', default=False)
    parser.add_argument('--max-q-tolerance', type=int, default=10)
    parser.add_argument('--full-data-for-q-update', action='store_true', default=False)
    parser.add_argument('--return-in-remaining', action='store_true', default=False)

    parser.add_argument('--use-base-traj-for-q', action='store_true', default=False)

    # parser.add_argument('--full-tt-split', action='store_true', default=False, 
    #     help="almost separates obs from train and test. If use-base is off - separates fully")
        
    # parser.add_argument('--new-full-tt-split', action='store_true', default=False, 
    #     help="separates obs from train and test. test taken only from base.")
    parser.add_argument('--split-type', type=str, default='standart')

    parser.add_argument('--logstd-stopgrad', action='store_true', default=False, 
        help="stops gradient for entropy with Q-updates")

    parser.add_argument('--equal-importance-ploss', action='store_true', default=False, 
        help="calculate policy loss when returns from q-function are counted as equals to the env returns")

    args = parser.parse_args()
    return args


args = dict(
        seed=777,
        gamma=.99,
        lambda_=1., # gae lambda coef
        lr=2e-3,
        linear_decay=True,
        value_loss_coef=.4,
        # entropy_coef=0.,
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

    if cmd_args.type in possible_types:
        args['type'] = cmd_args.type
    else:
        raise NotImplementedError

    args['K'] = cmd_args.K
    args['L'] = cmd_args.L
    args['M'] = cmd_args.M

    # if args['type']=='standart':
    #     args['K'] = 0
    #     args['L'] = 0
    #     args['M'] = 0

    if args['type']=='K-rollouts':
        args['M'] = 1

    args['num_timesteps'] = cmd_args.num_timesteps
    args['hidden_sizes'] = tuple(int(x) for x in cmd_args.layers.split('-'))
    if cmd_args.q_layers is not None:
        args['q_hidden_sizes'] = tuple(int(x) for x in cmd_args.q_layers.split('-'))
    else:
        args['q_hidden_sizes'] = args['hidden_sizes']
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

    if cmd_args.num_updates is not None:
        args['num_timesteps'] = cmd_args.num_updates * args['num_envs'] * args['num_steps']

    args['num_workers'] = cmd_args.num_workers
    args['split_between_devices'] = cmd_args.split_between_devices

    if cmd_args.q_updates in possible_q_updates:
        args['q_updates'] = cmd_args.q_updates
        if cmd_args.q_updates == 'none':
            args['q_updates'] = None
    else:
        raise NotImplementedError

    if cmd_args.policy_type in possible_policy_types:
        args['policy_type'] = cmd_args.policy_type
    else:
        raise NotImplementedError

    # args['q_loss_coef'] = cmd_args.q_loss_coef
    args['eval_with_q'] = cmd_args.eval_with_q
    args['init_log_std'] = cmd_args.init_log_std

    if not cmd_args.use_samples_for_log_update:
        raise NotImplementedError
    if cmd_args.split_type not in possible_split_types:
        raise NotImplementedError

    args['train_constants'] = dict(
        value_loss_coef=args['value_loss_coef'], 
        entropy_coef=cmd_args.entropy_coef, 
        normalize_advantages=args['normalize_advantages'], 
        q_updates=args['q_updates'],
        q_loss_coef=cmd_args.q_loss_coef,
        alpha=cmd_args.alpha,
        gamma=cmd_args.gamma,
        lambda_=args['lambda_'],
        M=args['M'],
        type=args['type'],
        gradstop=cmd_args.gradstop,
        entropy=cmd_args.entropy,
        q_targets=cmd_args.q_targets,
        q_loss_multiplier=cmd_args.q_loss_multiplier,
        qf_update_batch_size=cmd_args.qf_update_batch_size,
        qf_update_epochs=cmd_args.qf_update_epochs,
        qf_test_ratio=cmd_args.qf_test_ratio,
        use_samples_for_log_update=cmd_args.use_samples_for_log_update,
        use_q_tolerance=cmd_args.use_q_tolerance,
        max_q_tolerance=cmd_args.max_q_tolerance,
        full_data_for_q_update=cmd_args.full_data_for_q_update,
        return_in_remaining=cmd_args.return_in_remaining,
        K=args['K'],
        logstd_stopgrad=cmd_args.logstd_stopgrad,
        equal_importance_ploss=cmd_args.equal_importance_ploss,
    )

    args['sampling_type'] = cmd_args.sampling_type
    args['sampling_prob_temp'] = cmd_args.sampling_prob_temp
    args['n_samples'] = cmd_args.n_samples
    args['ignore_original_trajectory'] = cmd_args.ignore_original_trajectory
    args['km_determenistic'] = cmd_args.km_determenistic

    args['negative_sampling'] = cmd_args.negative_sampling
    args['use_base_traj_for_q'] = cmd_args.use_base_traj_for_q
    # args['full_tt_split'] = cmd_args.full_tt_split
    # args['new_full_tt_split'] = cmd_args.new_full_tt_split

    # if (cmd_args.new_full_tt_split and cmd_args.full_tt_split):
    #     raise Exception
    args['split_type'] = cmd_args.split_type
    return args

args = update(args, cmd_args)

if __name__=='__main__':
    print(args)
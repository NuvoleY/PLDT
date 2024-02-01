import os

import torch
import argparse
from prompt_lstm.model import PLDT
from prompt_lstm.trainer import Trainer
from prompt_lstm.utils import get_env
from prompt_lstm.utils import get_prompt_batch, get_prompt, get_batch
from prompt_lstm.utils import process_total_data_mean, load_data_prompt, process_info
from prompt_lstm.utils import eval_episodes
import os


def experiment_env(
        variant
):
    device = variant['device']
    cur_dir = os.getcwd()
    data_save_path = os.path.join(cur_dir, 'envs')

    train_env_name = args.env
    test_env_name = args.env
    info, envs = get_env(train_env_name, device)
    test_info, test_env = get_env(test_env_name,  device)

    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)
    mode = variant.get('mode', 'normal')
    dataset_mode = variant['dataset_mode']

    trajectories, prompt_trajectories = load_data_prompt(train_env_name, data_save_path, args)

    test_trajectories, test_prompt_trajectories = load_data_prompt(test_env_name, data_save_path, args)

    total_traj = trajectories + test_trajectories
    total_state_mean, total_state_std= process_total_data_mean(total_traj, mode)
    variant['total_state_mean'] = total_state_mean
    variant['total_state_std'] = total_state_std

    print_ = True
    info = process_info(train_env_name, trajectories, info, mode, dataset_mode, pct_traj, variant, print_)
    print_ = False
    test_info = process_info(test_env_name, test_trajectories, test_info, mode, dataset_mode, pct_traj, variant, print_)

    state_dim = test_env.observation_space.shape[0]
    act_dim = test_env.action_space.shape[0]

    obs_upper_bound = float(envs.observation_space.high[0])
    obs_lower_bound = float(envs.observation_space.low[0])

    model = PLDT(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=1000,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
        obs_upper_bound = obs_upper_bound,
        obs_lower_bound = obs_lower_bound
    )
    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    env_name = train_env_name
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        # get_batch=get_batch(Data_Augmentation(train_env_name, data_save_path), info[env_name], variant),
        get_batch=get_batch(trajectories, info[env_name], variant),
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
        eval_fns=None,
        get_prompt=get_prompt(prompt_trajectories, info[env_name], variant),
        get_prompt_batch=get_prompt_batch(trajectories, prompt_trajectories, info, variant, train_env_name)
    )
    if args.no_prompt:
        save_dir = f'{args.env}'
    else:
        save_dir = f'test'  #
    output_dir = os.path.join(f'./results/{args.env}_{dataset_mode}', save_dir)
    os.makedirs(output_dir, exist_ok=True)

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(
            num_steps=variant['num_steps_per_iter'],
            no_prompt=args.no_prompt
            )

        if iter % args.test_eval_interval == 0:
            test_eval_logs = trainer.eval_iteration(
                get_prompt, test_prompt_trajectories,
                eval_episodes, test_env_name, test_info, variant, test_env, iter_num=iter + 1,
                print_logs=True, no_prompt=args.no_prompt, group='test')
            outputs.update(test_eval_logs)

            if iter == 0:
                _basic_columns = ['iter']
                _record_values = [iter+1]
                for k, v in test_eval_logs.items():
                    _basic_columns.append(k)
                    _record_values.append(v)
                with open(os.path.join(output_dir, "log_test.txt"), "w") as f:
                    print("\t".join(_basic_columns), file=f)
                with open(os.path.join(output_dir, "log_test.txt"), "a+") as f:
                    print("\t".join(str(x) for x in _record_values), file=f)
            else:
                _record_values = [iter+1]
                for v in test_eval_logs.values():
                    _record_values.append(v)
                with open(os.path.join(output_dir, "log_test.txt"), "a+") as f:
                    print("\t".join(str(x) for x in _record_values), file=f)

        if iter % args.train_eval_interval == 0:
            train_eval_logs = trainer.eval_iteration(
                get_prompt, prompt_trajectories,
                eval_episodes, train_env_name, info, variant, envs, iter_num=iter + 1,
                print_logs=True, no_prompt=args.no_prompt, group='train')
            outputs.update(train_eval_logs)

            _record_values = [iter + 1]
            for v in train_eval_logs.values():
                _record_values.append(v)
            with open(os.path.join(output_dir, "log_train.txt"), "a+") as f:
                print("\t".join(str(x) for x in _record_values), file=f)
        outputs.update({"global_step": iter})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset_mode', type=str, default='expert')
    parser.add_argument('--prompt-length', type=int, default=5)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--no-prompt', action='store_true', default=False)
    parser.add_argument('--no_state_normalize', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000) # 10000*(number of environments)
    parser.add_argument('--num_eval_episodes', type=int, default=50)
    parser.add_argument('--max_iters', type=int, default=6000)
    parser.add_argument('--num_steps_per_iter', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_eval_interval', type=int, default=300)
    parser.add_argument('--test_eval_interval', type=int, default=30)

    args = parser.parse_args()
    experiment_env(variant=vars(args))
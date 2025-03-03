import gym
import numpy as np
import torch
import wandb

import argparse
import json
import pickle
import random
import sys
import os


sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/training")
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/testing")
from utils import AddCostToRewardEnv
from visualize import plot_waymo_vs_pred
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg, evaluate_episode_rtg_waymo
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

from metadrive.policy.replay_policy import ReplayEgoCarPolicy, PMKinematicsEgoPolicy
from metadrive.policy.env_input_policy import EnvInputHeadingAccPolicy
WAYMO_SAMPLING_FREQ = 10
acc_scale = 1

import pickle


    


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['param_set_name']#variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'


    # env = gym.make('Hopper-v3')
    env = AddCostToRewardEnv(
    {
        "manual_control": False,
        "no_traffic": False,
        "agent_policy": PMKinematicsEgoPolicy,
        "waymo_data_directory":variant['pkl_dir'],
        "case_num": 50000,
        "start_seed": 0,
        "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
        "use_render": False,
        "reactive_traffic": False,
                "vehicle_config": dict(
               # no_wheel_friction=True,
               lidar=dict(num_lasers=80, distance=50, num_others=4), # 120
               lane_line_detector=dict(num_lasers=12, distance=50), # 12
               side_detector=dict(num_lasers=20, distance=50) # 160
               ),
    }
    )
    print("building test set")
    test_config = env.config.copy()
    
    set_separate_validation_set = True
    if set_separate_validation_set: 
        test_config.update({
            "case_num": 100,
            "start_seed":0,
        })

    test_env = AddCostToRewardEnv(test_config)

    max_ep_len = 90
    env_targets = [400,1000] # evaluation conditioning targets
    scale = 100.  # normalization for rewards/returns

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    variant['max_ep_len'] = max_ep_len
    variant['scale'] = scale
    variant['state_dim'] = state_dim
    variant['act_dim'] = act_dim

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug
        ckpt_path = wandb.run.dir
        print("[DT] logging data to "+ ckpt_path)
    else:
        ckpt_path = None



    if model_type == 'bc':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    
    pkl_list = os.listdir(variant['dataset_dir'])
    
    trajectories = []
    # 
    for pkl_fn in pkl_list:
        with open(os.path.join(variant['dataset_dir'], pkl_fn), 'rb') as f:
            try:
                temp = pickle.load(f)
                trajectories.extend(temp)
            except:
                print("........ skipping "+ pkl_fn +" ........")
    
    trajectories = sorted(trajectories, key=lambda x: x['seed'])

    # report info about the waymo data collected:
    # print('=' * 50)
    # print(f'reward/cost info about the waymo data: ')
    # print(f'lambda = 10')
    # print(f'avg. reward per expert trajectory {np.mean([path['rewards'] for path in trajectories])}')
    # print(f'avg. cost per expert trajectory {np.mean([path['cost'] for path in trajectories])}')
    # print('=' * 50)
    # import pdb; pdb.set_trace()

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    ill_seed = []
    for path in trajectories:
        # print("path['rewards'].shape", path['rewards'].shape)
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        if path['rewards'].sum() > 0:

            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        else:
            # print("path['rewards'].sum() < -250: "+ str(path['seed']))
            ill_seed.append([path['seed'],path['rewards'].sum()] )
            
    trajectories = [d for d in trajectories if d.get("seed") not in np.array(ill_seed)[:,0]]
    # print("path['rewards'].sum() < 0: ", np.array(ill_seed) )
    np.save(os.path.join(variant['dataset_dir'], "ill_seed.npy"), np.array(ill_seed), allow_pickle=True)
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    log_stats = {
        "obs_mean": state_mean.tolist(),
        "obs_std": state_std.tolist(),
        "reward_scale": scale,
        "target_return": env_targets[0],
    }
    if ckpt_path:
        stats_path = os.path.join(ckpt_path, 'obs_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(log_stats, f, indent=2)

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))

            # xinyi: normalize acc from -5,5 to -1,1 # not any more
            ac = traj['actions'][si:si + max_len].reshape(1, -1, act_dim)
            # ac[0, :, 1] /= acc_scale 
            a.append(ac)
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device) # batchsize * K * obs_dim
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device) # batchsize * K * act_dim
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask
    
    fig_dir = "/home/xinyi/src/decision-transformer/gym/figs/training"
    fig_subdir = os.path.join(fig_dir, variant['param_set_name'])
    if not os.path.isdir(fig_subdir):
        os.makedirs(fig_subdir)

    def eval_episodes(target_rew):
        def fn(model):
            successes, returns, lengths = [], [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        # ret, length, is_success = evaluate_episode_rtg(
                        ret, length, is_success, seed =  evaluate_episode_rtg_waymo(
                            test_env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            rew_scale=scale,
                            acc_scale= acc_scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            save_fig_dir= fig_subdir
                            # save_fig_dir=""
                            )
                    else:
                        ret, length, is_success = evaluate_episode(
                            test_env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
                successes.append(is_success)
            return {
                f'target_{target_rew}_success_rate': np.mean(successes),
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim, 
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            action_tanh=False,
            hidden_size=variant['embed_dim'], # default 128
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            ckpt_path=ckpt_path,
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug
    

    print("variant['max_iters'] = ", variant['max_iters'], "variant['num_eval_episodes'] =", variant['num_eval_episodes'])
    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #### interested params ######
    
    parser.add_argument('--param_set_name', type=str, default='default-50000')
    parser.add_argument('--K', type=int, default= 20) # K =20 decides the dependent time window
    parser.add_argument('--batch_size', type=int, default=512) # increase to stablize, as well as dataset 
    parser.add_argument('--n_head', type=int, default=1) # could be devided by 128
    parser.add_argument('--warmup_steps', type=int, default=10000) #10000
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.25) #0.1
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4) 
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    
    ###############################

    ######### Test condition ######
    
    parser.add_argument('--num_eval_episodes', type=int, default=20)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--num_steps_per_iter', type=int, default=2000) # 5000
    
    ###############################
    parser.add_argument('--env', type=str, default='waymo')
    parser.add_argument('--dataset_dir', type=str, default='/home/xinyi/src/data/metadrive/dt_pkl/waymo_n_50000_lam_1_eps_10')  
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--pkl_dir', type=str, default='/home/xinyi/src/data/metadrive/pkl_9/')
    
    parser.add_argument('--pct_traj', type=float, default=1.) 
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128) 
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)
    # parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    
    args = parser.parse_args()

    experiment('metadrive-gym', variant=vars(args))

    
    

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

from metadrive.policy.replay_policy import PMKinematicsEgoPolicy


sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/training")
from utils import AddCostToRewardEnv
import matplotlib.pyplot as plt
from stable_baselines3.js_sac import utils as js_utils
from evaluate_episodes import evaluate_episode, evaluate_episode_rtg_waymo

WAYMO_SAMPLING_FREQ = 10


import pickle



def eval(model, model_type,
       num_eval_episodes, test_env,  state_dim, act_dim, 
       max_ep_len, rew_scale, acc_scale, target_rew, mode, state_mean, state_std, device):
    successes, returns, lengths, success_seeds = [], [], [], []
    for _ in range(num_eval_episodes):
        with torch.no_grad():
            if model_type == 'dt':
                ret, length, is_success, seed = evaluate_episode_rtg_waymo(
                    test_env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    rew_scale=rew_scale,
                    acc_scale = acc_scale,
                    target_return=target_rew/rew_scale,
                    mode=mode,
                    state_mean=state_mean,
                    state_std=state_std,
                    device=device,
                    save_fig_dir= "/home/xinyi/src/decision-transformer/gym/figs/guide_only_dt_training"
                )
                if is_success:
                    success_seeds.append(seed)
            else:
                ret, length, is_success = evaluate_episode(
                    test_env,
                    state_dim,
                    act_dim,
                    model,
                    max_ep_len=max_ep_len,
                    target_return=target_rew/rew_scale,
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
        f'success_seeds': success_seeds.sort()
    }


if __name__== "__main__":

    # import model
    expert_model_dir = '/home/xinyi/src/decision-transformer/gym/wandb/run-20230823_230743-3s6y7mzy'
    num_scenarios = 100
    loaded_stats = js_utils.load_demo_stats(
            path=expert_model_dir
        )
    obs_mean, obs_std, reward_scale, target_return = loaded_stats
    model = js_utils.load_transformer(
        model_dir=expert_model_dir,
        device = 'cpu'
    )

    test_env = AddCostToRewardEnv(
        {
            "manual_control": False,
            "no_traffic": False,
            "agent_policy":PMKinematicsEgoPolicy,
            "waymo_data_directory":'/home/xinyi/src/data/metadrive/pkl_9',
            "case_num": num_scenarios,
            "physics_world_step_size": 1/WAYMO_SAMPLING_FREQ, # have to be specified each time we use waymo environment for training purpose
            "use_render": False,
            'start_seed': 0,
            "horizon": 90/5,
            "reactive_traffic": False,
                    "vehicle_config": dict(
                # no_wheel_friction=True,
                lidar=dict(num_lasers=80, distance=50, num_others=4), # 120
                lane_line_detector=dict(num_lasers=12, distance=50), # 12
                side_detector=dict(num_lasers=20, distance=50)) # 160,
        },    
    )

    print(
        eval(model, 'dt',
       num_eval_episodes=100, test_env=test_env,  state_dim=145, act_dim=2, 
       max_ep_len=90, rew_scale=100, acc_scale=5., target_rew=400, mode='normal', state_mean=obs_mean, state_std=obs_std, device='cpu')
    )
    

    


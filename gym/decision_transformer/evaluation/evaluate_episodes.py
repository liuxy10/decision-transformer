import numpy as np
import torch
import sys
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/training")
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/data_processing")
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/testing")
import matplotlib.pyplot as plt
import os



def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, info = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length, info['arrive_dest']


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, info = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length, info['arrive_dest']

def evaluate_episode_rtg_waymo(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        save_fig_dir = "",
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0


    from collect_h5py_from_pkl import get_current_ego_trajectory_old
    seed = env.engine.global_random_seed

    #recorded ts, position, velocity, acc, heading from waymo
    ts, pos_rec, vel_rec, acc_rec, heading_rec, heading_rate_rec = get_current_ego_trajectory_old(env,seed)
    speed_rec = np.linalg.norm(vel_rec, axis = 1)
    
    
    pos_pred = np.zeros_like(pos_rec)
    action_pred = np.zeros_like(pos_rec) # default diff action version
    actual_heading = np.zeros_like(ts)
    actual_speed= np.zeros_like(ts)
    actual_rew = np.zeros_like(ts)
    cum_rew, cum_cost = 0,0

    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, info = env.step(action)

        

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        actual_heading[t] = env.engine.agent_manager.active_agents['default_agent'].heading_theta
        actual_speed[t] = np.array(env.engine.agent_manager.active_agents['default_agent'].speed/3.6)
        pos_pred [t,:] = np.array(env.engine.agent_manager.active_agents['default_agent'].position)
        action_pred [t,:] = action
        actual_rew[t] = reward
        cum_rew += reward
        cum_cost += info['cost']




        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)
        episode_return += reward
        episode_length += 1

        if done:
            actual_heading[t:] = None
            actual_speed[t:] = None
            break

    plot_comparison = True
    action_pred = np.array(action_pred)
    if plot_comparison:
        pos_pred = np.array(pos_pred)
        fig, axs = plt.subplots(2, 2)
        md_name = 'DT'
        axs[0,0].plot(ts, action_pred[:,1], label = md_name +' pred acc')
        axs[0,0].plot(ts, acc_rec, label = 'waymo acc' )
        axs[1,0].plot(ts, actual_heading, label = md_name +' actual heading' )
        axs[1,0].plot(ts, heading_rec, label = 'waymo heading')
        axs[0,1].plot(ts, actual_speed, label = md_name+' actual speed' )
        axs[0,1].plot(ts, speed_rec, label = 'waymo speed')
        axs[1,1].plot(ts, actual_rew, label = md_name+' actual reward' )
        # axs[1,1].plot(ts, rew_rec, label = 'waymo reward')
        for i in range(2):
            for j in range(2):
                axs[i,j].legend()
                axs[i,j].set_xlabel('time')
        axs[0,0].set_ylabel('acceleration')
        axs[1,0].set_ylabel('heading')
        axs[0,1].set_ylabel('speed')
        axs[1,1].set_ylabel('reward')
        # plt.title("recorded action vs test predicted action")
        if len(save_fig_dir) > 0:
            if not os.path.isdir(save_fig_dir):
                os.makedirs(save_fig_dir)
            plt.savefig(os.path.join(save_fig_dir, "seed_"+str(seed)+".jpg"))
        else:
            plt.show()

    return episode_return, episode_length, info['arrive_dest']

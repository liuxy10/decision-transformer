import numpy as np
import torch
import sys
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/training")
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/data_processing")
sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/testing")
import matplotlib.pyplot as plt
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image



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
    
    
    actual_pos = np.zeros_like(pos_rec)
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
        if abs(action[1]) >1:
            print("[evaluate_episode] ~~~~~~~~~~~~~~~~~~~ means that DT generate actions larger than 1~~~~~~~~~~~~~~")
        action = action.detach().cpu().numpy()

        state, reward, done, info = env.step(action)

        

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        actual_heading[t] = env.engine.agent_manager.active_agents['default_agent'].heading_theta
        actual_speed[t] = np.array(env.engine.agent_manager.active_agents['default_agent'].speed/3.6)
        actual_pos [t,:] = np.array(env.engine.agent_manager.active_agents['default_agent'].position)
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
            actual_heading[t+1:] = None
            actual_speed[t+1:] = None
            action_pred[t+1:] = None
            actual_rew[t+1:] = None
            actual_pos[t+1:,:] = None

            break

    plot_comparison = True
    action_pred = np.array(action_pred)
    if plot_comparison:
        plot_states_compare(ts, 
                   action_pred, acc_rec, 
                   actual_speed, speed_rec, 
                   pos_rec, actual_pos, 
                   actual_heading, heading_rec, 
                   actual_rew,
                   save_fig_dir,
                   seed)

    return episode_return, episode_length, info['arrive_dest'], env.engine.global_random_seed

def plot_states_compare(ts, 
                   action_pred, acc_rec, 
                   actual_speed, speed_rec, 
                   pos_rec, actual_pos, 
                   actual_heading, heading_rec, 
                   actual_rew,
                   save_fig_dir, seed):
    actual_pos = np.array(actual_pos)
    fig, axs = plt.subplots(2, 2, figsize = (12,12), gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]})
    md_name = 'DT'
    axs[0,0].plot(ts, action_pred[:,1], label = md_name +' pred acc')
    axs[0,0].plot(ts, acc_rec, label = 'waymo acc' )
    # axs[1,0].plot(ts, actual_heading, label = md_name +' actual heading' )
    # axs[1,0].plot(ts, heading_rec, label = 'waymo heading')
    axs[1,0].set_aspect('equal')
    plot_car(axs[1,0], actual_pos[:,0], actual_pos[:,1], actual_heading, label = md_name)
    plot_car(axs[1,0], pos_rec[:,0], pos_rec[:,1], heading_rec, label = "waymo")
    plot_dest_range(axs[1,0], pos_rec[-1,:], 5)
    axs[0,1].plot(ts, actual_speed, label = md_name+' actual speed' )
    axs[0,1].plot(ts, speed_rec, label = 'waymo speed')
    axs[1,1].plot(ts, actual_rew, label = md_name+' actual reward' )
    
    # axs[1,1].plot(ts, rew_rec, label = 'waymo reward')
    for i in range(2):
        for j in range(2):
            axs[i,j].legend()
            axs[i,j].set_xlabel('time')
            if (i,j) != (1,0):
                axs[i,j].set_xlim([0,9])
            else:
                x_mid, y_mid = pos_rec[45,0],pos_rec[45,1]
                w = max(max(6, max(np.ptp(pos_rec[:,0]),np.ptp(pos_rec[:,1]))),
                        np.max(abs(actual_pos[:,0] - x_mid)))
                axs[i,j].set_xlim([x_mid - w, x_mid + w])
                axs[i,j].set_ylim([y_mid - w, y_mid + w])
    
    axs[0,0].set_ylabel('acceleration')
    axs[0,1].set_ylabel('speed')
    axs[1,1].set_ylabel('reward')
    axs[1,0].set_xlabel('X Position')
    axs[1,0].set_ylabel('Y Position')
    axs[1,0].set_title('Trajectories')
    # plt.title("recorded action vs test predicted action")
    if len(save_fig_dir) > 0:
        if not os.path.isdir(save_fig_dir):
            os.makedirs(save_fig_dir)
        plt.savefig(os.path.join(save_fig_dir, "seed_"+str(seed)+".jpg"))
    else:
        plt.show()

def plot_dest_range(ax, center, radius):
    circle = plt.Circle(center, radius, fill=False, edgecolor='red')
    ax.add_patch(circle)

def plot_car(ax, xs, ys, headings, label):
    
    
    if label == "waymo":
        car_icon_path = '/home/xinyi/src/decision-transformer/gym/car_red.png'
        car_icon = plt.imread(car_icon_path)
        ax.plot(xs, ys, label = label, color = "red")
    else:
        car_icon_path = '/home/xinyi/src/decision-transformer/gym/car_blue.png'
        car_icon = plt.imread(car_icon_path)
        ax.plot(xs, ys, label = label, color = "blue")
    for i in range(xs.shape[0]):
        if i % 10 == 0 and ~np.isnan(xs[i]):
            # Plot car icon
            transparency = min(1, max(0, 1/2 + i / 180))
            
            # Rotate the car icon based on heading
            rotated_car_icon = rotate_image(car_icon_path, -90 + headings[i] * 180 /np.pi)
            imagebox = OffsetImage(rotated_car_icon, zoom=0.05, alpha =transparency )
            ab = AnnotationBbox(imagebox, (xs[i], ys[i]), frameon=False)
            ax.add_artist(ab)
    

def rotate_image(path, angle):
    """
    Rotate the given image by the given angle while preventing clipping.
    """
    pil_image = Image.open(path)
  
    # Convert PIL image to NumPy array
    image_array = np.array(pil_image)

    # Calculate the new dimensions of the canvas
    height, width = image_array.shape[:2]
    new_height = height * 3
    new_width = width * 3

    # Create a larger canvas
    canvas = np.zeros((new_height, new_width, image_array.shape[2]), dtype=image_array.dtype)

    # Calculate the center of the original and rotated images
    center_x = width // 2
    center_y = height // 2
    new_center_x = new_width // 2
    new_center_y = new_height // 2

    # Calculate the top-left corner of the rotated image on the canvas
    top_left_x = new_center_x - center_x
    top_left_y = new_center_y - center_y

    # Place the original image on the canvas
    canvas[top_left_y:top_left_y+height, top_left_x:top_left_x+width, :] = image_array

    # Rotate the canvas
    rotated_canvas = Image.fromarray(canvas).rotate(angle, resample=Image.BILINEAR, expand=1)

    return np.array(rotated_canvas)
    




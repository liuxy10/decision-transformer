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


def plot_states_compare(ts, 
                   action_pred, acc_rec, 
                   actual_speed, speed_rec, 
                   pos_rec, actual_pos, 
                   actual_heading, heading_rec, 
                   actual_rew,
                   save_fig_dir, seed, succeed):
    actual_pos = np.array(actual_pos)
    fig_states, axs = plt.subplots(2, 2, figsize = (12,8))
    fig_bv, ax = plt.subplots(1,1, figsize = (8,8)) #, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]} )
    md_name = 'DT'
    colors = {'waymo': 'blue', 'DT': 'red'}
    axs[0,0].plot(ts, acc_rec, color = colors['waymo'],  label = 'waymo acc' )
    axs[0,0].plot(ts, action_pred[:,1], color = colors[md_name], label = md_name +' pred acc')
    axs[1,0].plot(ts, actual_heading, color = colors[md_name],label = md_name +' actual heading' )
    axs[1,0].plot(ts, heading_rec,  color = colors['waymo'], label = 'waymo heading')
    
    axs[0,1].plot(ts, actual_speed, color = colors[md_name], label = md_name+' actual speed' )
    axs[0,1].plot(ts, speed_rec, color = colors['waymo'], label = 'waymo speed')
    axs[1,1].plot(ts, actual_rew, color = colors[md_name],label = md_name+' actual reward' )
    # axs[1,1].plot(ts, rew_rec, label = 'waymo reward')
    
    ax.set_aspect('equal')
    plot_car(ax, actual_pos[:,0], actual_pos[:,1], actual_heading, label = md_name)
    plot_car(ax, pos_rec[:,0], pos_rec[:,1], heading_rec, label = "waymo")
    plot_dest_range(ax, pos_rec[-1,:], 5)

    
    for i in range(2):
        for j in range(2):
            axs[i,j].legend()
            axs[i,j].set_xlabel('time')
            axs[i,j].set_xlim([0,9])

    
    x_mid, y_mid = pos_rec[45,0],pos_rec[45,1]
    w = max(max(6, max(np.ptp(pos_rec[:,0]),np.ptp(pos_rec[:,1]))),
            np.max(abs(actual_pos[:,0] - x_mid)))
    ax.set_xlim([x_mid - w, x_mid + w])
    ax.set_ylim([y_mid - w, y_mid + w])
    ax.legend()
          
    
    axs[0,0].set_ylabel('acceleration')
    axs[0,1].set_ylabel('speed')
    axs[1,1].set_ylabel('reward')
    axs[1,0].set_ylabel('heading')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Trajectories')
    
    # plt.title("recorded action vs test predicted action")
    if len(save_fig_dir) > 0:
        if not os.path.isdir(save_fig_dir):
            os.makedirs(save_fig_dir)
        fig_states.savefig(os.path.join(save_fig_dir, "seed_"+str(seed)+f"_states_{succeed}.jpg"))
        fig_bv.savefig(os.path.join(save_fig_dir, "seed_"+str(seed)+f"_birdview_{succeed}.jpg"))

    else:
        plt.show()

def plot_dest_range(ax, center, radius):
    circle = plt.Circle(center, radius, fill=False, edgecolor='blue')
    ax.add_patch(circle)

def plot_car(ax, xs, ys, headings, label):
    
    
    if label == 'DT':
        car_icon_path = '/home/xinyi/src/decision-transformer/gym/car_red.png'
        ax.plot(xs, ys, label = label, color = "red")
    elif label == 'waymo':
        car_icon_path = '/home/xinyi/src/decision-transformer/gym/car_blue.png'
        # car_icon = plt.imread(car_icon_path)
        ax.plot(xs, ys, label = label, color = "blue")
    elif label == 'traffic':
        car_icon_path = '/home/xinyi/src/decision-transformer/gym/car_purple.png'
        # car_icon = plt.imread(car_icon_path)
        ax.plot(xs, ys, label = label, color = "purple")

    for i in range(xs.shape[0]):
        if i % 10 == 0 and ~np.isnan(xs[i]):
            # Plot car icon
            transparency = min(1, max(0, 1/2 + i / 180))
            
            # Rotate the car icon based on heading
            rotated_car_icon = rotate_image(car_icon_path, -90 + headings[i] * 180 /np.pi)
            imagebox = OffsetImage(rotated_car_icon, 
                                    zoom= 0.05, #0.05, 
                                    alpha =transparency )
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



if __name__ == "__main__":

    import pickle
    import sys
    sys.path.append("/home/xinyi/src/safe-sb3/examples/metadrive/data_processing")
    from waymo_utils import *
    
    pickle_file_path = "/home/xinyi/src/data/metadrive/pkl_9/0.pkl"

    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
    # read_waymo_data(pickle_file_path)
    fig_bv, ax = plt.subplots(1,1, figsize = (12,12)) #, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1]} )
    draw_waymo_map(data)
    

    for key in data["tracks"]:
        car_states = data["tracks"][key]["state"] # x: 0, y:1, heading:6 of certain traffic
        xs, ys, headings = car_states[:,0],car_states[:,1], car_states[:,6]
        for a in [xs, ys, headings]:
            a[a==0] = np.nan # to replace the zeros in the traffic data
        plot_car(ax, xs, ys, headings, label = "traffic")

    plt.savefig("/home/xinyi/src/decision-transformer/gym/figs/map_visualization/map+car.png")
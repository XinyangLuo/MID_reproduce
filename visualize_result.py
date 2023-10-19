import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.bitmap import BitMap

from evaluation.visualization import visualize_prediction

def plot_ego_trajectory(ax, ego_trajectory, ego_timesteps, t, max_hl, ph):
    t_index = t - ego_timesteps[0]
    ax.plot(ego_trajectory[t_index - max_hl + 1:t_index + 1, 0], ego_trajectory[t_index - max_hl + 1:t_index + 1, 1], '--', c='#1f77b4', alpha = 0.6)
    ax.plot(ego_trajectory[t_index + 1:t_index + 1 + ph, 0], ego_trajectory[t_index + 1:t_index + 1 + ph, 1], c='#1f77b4', alpha = 0.6)
    circle = plt.Circle((ego_trajectory[t_index, 0], ego_trajectory[t_index, 1]), 0.3, facecolor='g', edgecolor='k', lw=0.5, zorder=3)
    ax.add_artist(circle)

def visualize_timestep_prediction(t, predictions, ego_trajectory, ego_timesteps, dt, max_hl, ph):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    visualize_prediction(ax, {t: predictions}, dt, max_hl=max_hl, ph=ph)
    plot_ego_trajectory(ax, ego_trajectory, ego_timesteps, t, max_hl, ph)
    ax.set_title("timestep " + str(t))
    plt.show()

def visualize_timestep_prediction_map(t, predictions, ego_trajectory, ego_timesteps, dt, max_hl, ph, map_name, patch):
    nusc_map = NuScenesMap(dataroot='/home/yudie/dataset/nuscenes/full/', map_name=map_name)
    bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
    fig, ax = nusc_map.render_map_patch(patch, ['drivable_area'], figsize=(10, 10), bitmap=bitmap)
    visualize_prediction(ax, {t: predictions}, dt, max_hl=max_hl, ph=ph)
    plot_ego_trajectory(ax, ego_trajectory, ego_timesteps, t, max_hl, ph)
    plt.show()

def visualize_timesteps_prediction(predictions_dict, ego_trajectory, ego_timesteps, dt, max_hl, ph):
    fig = plt.figure(figsize=(12, 8))
    timesteps = list(predictions_dict.keys())
    col = len(timesteps) if len(timesteps) <= 4 else 4
    row = 1 if len(timesteps) <= 4 else (len(timesteps) - 1) // col + 1
    for i, t in enumerate(timesteps):
      ax = fig.add_subplot(row, col, i + 1)
      visualize_prediction(ax, {t: predictions_dict[t]}, dt, max_hl=max_hl, ph=ph)
      plot_ego_trajectory(ax, ego_trajectory, ego_timesteps, t, max_hl, ph)
      ax.set_title("timestep " + str(t))
    plt.show()

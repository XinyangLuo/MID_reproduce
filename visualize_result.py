import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.bitmap import BitMap

from evaluation.visualization import visualize_prediction

def plot_ego_trajectory(ax, ego_trajectory, max_hl):
    ax.plot(ego_trajectory[0: max_hl, 0], ego_trajectory[0: max_hl, 1], '--', c='#1f77b4', alpha = 0.6)
    ax.plot(ego_trajectory[max_hl + 1:, 0], ego_trajectory[max_hl + 1:, 1], c='#1f77b4', alpha = 0.6)
    circle = plt.Circle((ego_trajectory[max_hl, 0], ego_trajectory[max_hl, 1]), 0.3, facecolor='g', edgecolor='k', lw=0.5, zorder=3)
    ax.add_artist(circle)

def visualize_timestep_prediction(t, predictions, ego_positions, dt, max_hl, ph):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    visualize_prediction(ax, {t: predictions}, dt, max_hl=max_hl, ph=ph)
    plot_ego_trajectory(ax, ego_positions, max_hl)
    torch.save(ego_positions, 'ego.pt')
    ax.set_title("timestep " + str(t))
    plt.show()

def visualize_timestep_prediction_map(t, predictions, ego_positions, dt, max_hl, ph, map_name, patch):
    nusc_map = NuScenesMap(dataroot='/home/yudie/dataset/nuscenes/full/', map_name=map_name)
    bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')
    fig, ax = nusc_map.render_map_patch(patch, ['drivable_area'], figsize=(10, 10), bitmap=bitmap)
    visualize_prediction(ax, {t: predictions}, dt, max_hl=max_hl, ph=ph)
    plot_ego_trajectory(ax, ego_positions, max_hl)
    plt.show()

def visualize_timesteps_prediction(predictions_dict, ego_positions, dt, max_hl, ph):
    fig = plt.figure(figsize=(16, 8))
    timesteps = list(predictions_dict.keys())
    col = len(timesteps) if len(timesteps) <= 4 else 4
    row = 1 if len(timesteps) <= 4 else (len(timesteps) - 1) // col + 1
    for i, t in enumerate(timesteps):
      ax = fig.add_subplot(row, col, i + 1)
      visualize_prediction(ax, {t: predictions_dict[t]}, dt, max_hl=max_hl, ph=ph)
      plot_ego_trajectory(ax, ego_positions, max_hl)
      ax.set_title("timestep " + str(t))
    plt.show()

def visualize_node_prediction(pred_positions, pred_velocities, pred_accelerations, pred_jerks, ego_positions, index=0, dt=0.5):
    node_cmap = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    fig = plt.figure("node " + str(index), figsize=(16, 10))
    fig_kinematic = plt.figure("node " + str(index) + " kinematic", figsize=(16, 10))
    ts_velocities = np.linspace(0.0, dt*pred_velocities.shape[1], pred_velocities.shape[1])
    ts_accelerations = np.linspace(0.0, dt*pred_accelerations.shape[1], pred_accelerations.shape[1])
    ts_jerks = np.linspace(0.0, dt*pred_jerks.shape[1], pred_jerks.shape[1])
    for i in range(20):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.plot(pred_positions[i, :, 0], pred_positions[i, :, 1], '-o', c=node_cmap[index], alpha = 0.6)
        ax.plot(ego_positions[:, 0], ego_positions[:, 1], '-o', c='#1f77b4', alpha = 0.6)
        ax.legend(['node', 'ego'])
        ax.axis('equal')

        ax = fig_kinematic.add_subplot(4, 5, i + 1)
        ax.plot(ts_velocities, np.linalg.norm(pred_velocities[i], axis=1), '-o', alpha = 0.6)
        ax.plot(ts_accelerations, pred_accelerations[i], '-o', alpha = 0.6)
        ax.plot(ts_jerks, pred_jerks[i], '-o', alpha = 0.6)
        ax.legend(['vel', 'acc', 'jerk'])


from ..trajectory_utils import prediction_output_to_trajectories
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns

node_cmap = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      line_alpha=1.0,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=0.3,
                      batch_num=0,
                      kde=False):

    cmap = ['k', 'b', 'y', 'g', 'r']

    history_list = []
    future_list = []
    predictions_list = []
    for i, node in enumerate(histories_dict):
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]
        history_list.append(history)
        future_list.append(future)
        predictions_list.append(predictions)

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], 'k--')

        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 0.2
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            # ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
            #         color=cmap[node.type.value],
            #         linewidth=line_width, alpha=line_alpha)
            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                    color=node_cmap[i%9],
                    linewidth=line_width, alpha=line_alpha)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])

            # Current Node Position
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor='g',
                                edgecolor='k',
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

    ax.axis('equal')


def visualize_prediction(ax,
                         prediction_output_dict,
                         dt,
                         max_hl,
                         ph,
                         robot_node=None,
                         map=None,
                         **kwargs):

    prediction_dict, histories_dict, futures_dict = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    assert(len(prediction_dict.keys()) <= 1)
    if len(prediction_dict.keys()) == 0:
        return
    ts_key = list(prediction_dict.keys())[0]

    prediction_dict = prediction_dict[ts_key]
    histories_dict = histories_dict[ts_key]
    futures_dict = futures_dict[ts_key]

    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)
    plot_trajectories(ax, prediction_dict, histories_dict, futures_dict, *kwargs)

def plot_ego_positions(ax, ego_positions, max_hl):
    ax.plot(ego_positions[0: max_hl, 0], ego_positions[0: max_hl, 1], '--', c='#1f77b4', alpha = 0.6)
    ax.plot(ego_positions[max_hl + 1:, 0], ego_positions[max_hl + 1:, 1], c='#1f77b4', alpha = 0.6)
    circle = plt.Circle((ego_positions[max_hl, 0], ego_positions[max_hl, 1]), 0.3, facecolor='g', edgecolor='k', lw=0.5, zorder=3)
    ax.add_artist(circle)

def visualize_prediction_with_ego(t, predictions_dict, ego_positions, dt, max_hl, ph):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    visualize_prediction(ax, {t: predictions_dict}, dt, max_hl=max_hl, ph=ph)
    plot_ego_positions(ax, ego_positions, max_hl)
    ax.set_title("timestep " + str(t))

def visualize_prediction_with_map(t, predictions_dict, ego_positions, dt, max_hl, ph, nusc_map):
    patch = (ego_positions[:, 0].mean()-100.0, ego_positions[:, 1].mean()-100.0, ego_positions[:, 0].mean()+100.0, ego_positions[:, 1].mean()+100.0)
    fig, ax = nusc_map.render_map_patch(patch, ['drivable_area', 'lane'], figsize=(10, 10))
    visualize_prediction(ax, {t: predictions_dict}, dt, max_hl=max_hl, ph=ph)
    plot_ego_positions(ax, ego_positions, max_hl)
    ax.set_xlim(patch[0], patch[2])
    ax.set_ylim(patch[1], patch[3])

def visualize_prediction_with_derivations(predicted_positions, predicted_derivations, ego_positions, node_centreline_poses, index=0, dt=0.5):
    fig = plt.figure("node " + str(index), figsize=(16, 10))
    fig_kinematic = plt.figure("node " + str(index) + " kinematic", figsize=(16, 10))
    ts = np.linspace(0.0, dt*predicted_derivations[0].shape[1], predicted_derivations[0].shape[1])
    for i in range(20):
        ax = fig.add_subplot(4, 5, i + 1)
        ax.plot(predicted_positions[i, :, 0], predicted_positions[i, :, 1], '-o', c=node_cmap[index], alpha = 0.6)
        ax.plot(ego_positions[:, 0], ego_positions[:, 1], '-o', c='#1f77b4', alpha = 0.6)
        ax.plot(node_centreline_poses[:, 0], node_centreline_poses[:, 1], 'gray', alpha = 0.6)
        ax.legend(['node', 'ego', 'centreline'])
        ax.axis('equal')

        ax = fig_kinematic.add_subplot(4, 5, i + 1)
        if len(predicted_derivations) == 3:
            ax.plot(ts, predicted_derivations[0][i], '-o', alpha = 0.6)
            ax.plot(ts[:-1], predicted_derivations[1][i], '-o', alpha = 0.6)
            ax.plot(ts[:-2], predicted_derivations[2][i], '-o', alpha = 0.6)
            ax.legend(['vel', 'acc', 'jerk'])
        elif len(predicted_derivations) == 6:
            ax.plot(ts, predicted_derivations[0][i], '-o', alpha = 0.6)
            ax.plot(ts, predicted_derivations[1][i], '-o', alpha = 0.6)
            ax.plot(ts[:-1], predicted_derivations[2][i], '-o', alpha = 0.6)
            ax.plot(ts, predicted_derivations[3][i], '-o', alpha = 0.6)
            ax.plot(ts[:-1], predicted_derivations[4][i], '-o', alpha = 0.6)
            ax.plot(ts[:-2], predicted_derivations[5][i], '-o', alpha = 0.6)
            ax.legend(['vel', 'acc', 'jerk', 'angular_vel', 'angular_acc', 'angular_jerk'])   
import argparse
import dill
import os
from tqdm.auto import tqdm
from pyquaternion import Quaternion

import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from environment import Environment, Scene, Node, derivative_of

dt = 0.5 # annotate 2 Hz
scene_blacklist = [499, 515, 517]
data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 15},
            'y': {'mean': 0, 'std': 15},
            'norm': {'mean': 0, 'std': 15}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 4},
            'y': {'mean': 0, 'std': 4},
            'norm': {'mean': 0, 'std': 4}
        },
        'heading': {
            # 'x': {'mean': 0, 'std': 1},
            # 'y': {'mean': 0, 'std': 1},
            '°': {'mean': 0, 'std': np.pi},
            'd°': {'mean': 0, 'std': 1}
        }
    }
}

def process_pedestrian_data(x, y, dt):
    vx = derivative_of(x, dt)
    vy = derivative_of(y, dt)
    ax = derivative_of(vx, dt)
    ay = derivative_of(vy, dt)

    data_dict = {('position', 'x'): x,
                 ('position', 'y'): y,
                 ('velocity', 'x'): vx,
                 ('velocity', 'y'): vy,
                 ('acceleration', 'x'): ax,
                 ('acceleration', 'y'): ay}

    return pd.DataFrame(data_dict, columns=data_columns_pedestrian)

def process_vehicle_data(x, y, heading, dt):
    vx = derivative_of(x, dt)
    vy = derivative_of(y, dt)
    ax = derivative_of(vx, dt)
    ay = derivative_of(vy, dt)

    # v = np.stack((vx, vy), axis=-1)
    # v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
    # heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
    # heading_x = heading_v[:, 0]
    # heading_y = heading_v[:, 1]

    data_dict = {('position', 'x'): x,
                 ('position', 'y'): y,
                 ('velocity', 'x'): vx,
                 ('velocity', 'y'): vy,
                  ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                 ('acceleration', 'x'): ax,
                 ('acceleration', 'y'): ay,
                 ('acceleration', 'norm'): np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1),
                 #  ('heading', 'x'): heading_x,
                 #  ('heading', 'y'): heading_y,
                 ('heading', '°'): heading,
                 ('heading', 'd°'): derivative_of(heading, dt, radian=True)}

    return pd.DataFrame(data_dict, columns=data_columns_vehicle)

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc
    
    def rotate_node(node, alpha):
        if node.type == 'PEDESTRIAN':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            x, y = rotate_pc(np.array([x, y]), alpha)

            return Node(node_type=node.type, node_id=node.id, data=process_pedestrian_data(x, y, scene.dt), first_timestep=node.first_timestep)
        elif node.type == 'VEHICLE':
            x = node.data.position.x.copy()
            y = node.data.position.y.copy()

            heading = getattr(node.data.heading, '°').copy()
            heading += alpha
            heading = (heading + np.pi) % (2.0 * np.pi) - np.pi

            x, y = rotate_pc(np.array([x, y]), alpha)

            return Node(node_type=node.type, node_id=node.id, data=process_vehicle_data(x, y, heading, scene.dt), first_timestep=node.first_timestep,
                        non_aug_node=node)

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        scene_aug.nodes.append(rotate_node(node, alpha))
    
    scene_aug.ego_node = rotate_node(scene.ego_node, alpha)

    return scene_aug

def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug

def process_scene(ns_scene, env, nusc):
    scene_id = int(ns_scene['name'].replace('scene-', ''))
    data = pd.DataFrame(columns=['frame_id', 'type', 'node_id', 'x', 'y', 'z', 'heading'])
    ego_x, ego_y, ego_heading = [], [], []
    sample_token = ns_scene['first_sample_token']
    sample = nusc.get('sample', sample_token)
    frame_id = 0
    while sample['next']:
        annotation_tokens = sample['anns']
        for annotation_token in annotation_tokens:
            annotation = nusc.get('sample_annotation', annotation_token)
            category = annotation['category_name']
            if len(annotation['attribute_tokens']):
                attribute = nusc.get('attribute', annotation['attribute_tokens'][0])['name']
            else:
                continue

            if 'pedestrian' in category and not 'stroller' in category and not 'wheelchair' in category:
                our_category = env.NodeType.PEDESTRIAN
            elif 'vehicle' in category and 'bicycle' not in category and 'motorcycle' not in category and 'parked' not in attribute:
                our_category = env.NodeType.VEHICLE
            else:
                continue

            data.loc[len(data)] = pd.Series({'frame_id': frame_id,
                                             'type': our_category,
                                             'node_id': annotation['instance_token'],
                                             'x': annotation['translation'][0],
                                             'y': annotation['translation'][1],
                                             'z': annotation['translation'][2],
                                             'heading': Quaternion(annotation['rotation']).yaw_pitch_roll[0]})

        # Ego Vehicle (omit)

        sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        annotation = nusc.get('ego_pose', sample_data['ego_pose_token'])
        ego_x.append(annotation['translation'][0])
        ego_y.append(annotation['translation'][1])
        ego_heading.append(Quaternion(annotation['rotation']).yaw_pitch_roll[0])

        sample = nusc.get('sample', sample['next'])
        frame_id += 1

    if len(data.index) == 0:
        return None

    data.sort_values('frame_id', inplace=True)
    max_timesteps = data['frame_id'].max()

    # data['x'] = data['x'] - data['x'].mean()
    # data['y'] = data['y'] - data['y'].mean()

    scene = Scene(timesteps=max_timesteps + 1, dt=dt, name=str(scene_id), aug_func=augment)
    scene.ego_node = Node(node_type=env.NodeType.VEHICLE, node_id='ego', data=process_vehicle_data(np.array(ego_x), np.array(ego_y), np.array(ego_heading), scene.dt))

    # Generate Maps (omit)

    for node_id in pd.unique(data['node_id']):
        node_df = data[data['node_id'] == node_id]

        if node_df['x'].shape[0] < 2:
            continue

        if not np.all(np.diff(node_df['frame_id']) == 1):
            # print('Occlusion')
            continue  # TODO Make better

        node_values = node_df[['x', 'y']].values
        x = node_values[:, 0].astype(float)
        y = node_values[:, 1].astype(float)
        heading = node_df['heading'].values.astype(float)

        # Kalman filter (omit)

        if node_df.iloc[0]['type'] == env.NodeType.VEHICLE:
            node_data = process_vehicle_data(x, y, heading, scene.dt)
        else:
            node_data = process_pedestrian_data(x, y, scene.dt)
        
        node = Node(node_type=node_df.iloc[0]['type'], node_id=node_id, data=node_data)
        node.first_timestep = node_df['frame_id'].iloc[0]
        scene.nodes.append(node)
    return scene

def process_data(data_path, version, output_path):
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    splits = create_splits_scenes() # returns a mapping from split to scene names
    train_scene_names = splits['train' if 'mini' not in version else 'mini_train']
    test_scene_names = splits['val' if 'mini' not in version else 'mini_val']

    ns_scene_names = dict()
    ns_scene_names['train'] = train_scene_names
    ns_scene_names['test'] = test_scene_names

    for data_class in ['train', 'test']:
        env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization)
        attention_radius = dict()
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 10.0
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.VEHICLE)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.PEDESTRIAN)] = 20.0
        attention_radius[(env.NodeType.VEHICLE, env.NodeType.VEHICLE)] = 30.0

        env.attention_radius = attention_radius
        # env.robot_type = env.NodeType.VEHICLE
        scenes = []

        for ns_scene_name in tqdm(ns_scene_names[data_class]):
            ns_scene = nusc.get('scene', nusc.field2token('scene', 'name', ns_scene_name)[0]) # field2token return a list, use [0] get token
            scene_id = int(ns_scene['name'].replace('scene-', ''))
            if scene_id in scene_blacklist:  # Some scenes have bad localization
                continue

            scene = process_scene(ns_scene, env, nusc)
            if scene is not None:
                if data_class == 'train':
                    scene.augmented = list()
                    angles = np.arange(0, 360, 15)
                    for angle in angles:
                        scene.augmented.append(augment_scene(scene, angle))
                scenes.append(scene)

        print(f'Processed {len(scenes):.2f} scenes')

        env.scenes = scenes

        if len(scenes) > 0:
            mini_string = ''
            if 'mini' in version:
                mini_string = 'mini_'
            data_dict_path = os.path.join(output_path, 'nuscenes_' + mini_string + data_class + '.pkl')
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)
            print('Saved Environment!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    process_data(args.data_path, args.version, args.output_path)
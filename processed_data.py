import sys
import os
import numpy as np
import pandas as pd
import dill
import pickle

from environment import Environment, Scene, Node, derivative_of

# =========================
# Global config
# =========================
desired_max_time = 100
pred_indices = [2, 3]
state_dim = 4
frame_diff = 10
desired_frame_diff = 1
dt = 0.4

# =========================
# Standardization
# =========================
standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        }
    }
}

# =========================
# Utils
# =========================
def maybe_makedirs(path_to_create):
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


# =========================
# Data augmentation (rotation)
# =========================
def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha),  np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product(
        [['position', 'velocity'], ['x', 'y']]
    )

    scene_aug = Scene(
        timesteps=scene.timesteps,
        dt=scene.dt,
        name=scene.name
    )

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = np.asarray(node.data.position.x).copy()
        y = np.asarray(node.data.position.y).copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)

        data_dict = {
            ('position', 'x'): x,
            ('position', 'y'): y,
            ('velocity', 'x'): vx,
            ('velocity', 'y'): vy
        }

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node_new = Node(
            node_type=node.type,
            node_id=node.id,
            data=node_data,
            first_timestep=node.first_timestep
        )

        scene_aug.nodes.append(node_new)

    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


# =========================
# Output dir
# =========================
data_folder_name = 'processed_data_noise'
maybe_makedirs(data_folder_name)

data_columns = pd.MultiIndex.from_product(
    [['position', 'velocity'], ['x', 'y']]
)

# =========================
# ETH–UCY processing
# =========================
for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
    for data_class in ['train', 'val', 'test']:

        env = Environment(
            node_type_list=['PEDESTRIAN'],
            standardization=standardization
        )

        env.attention_radius = {
            (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 3.0
        }

        scenes = []
        data_dict_path = os.path.join(
            data_folder_name,
            f'{desired_source}_{data_class}.pkl'
        )

        for subdir, _, files in os.walk(
            os.path.join('raw_data', desired_source, data_class)
        ):
            for file in files:
                if not file.endswith('.txt'):
                    continue

                full_data_path = os.path.join(subdir, file)
                print('Processing', full_data_path)

                data = pd.read_csv(
                    full_data_path,
                    sep='\t',
                    header=None,
                    names=['frame_id', 'track_id', 'pos_x', 'pos_y']
                )

                data['frame_id'] = pd.to_numeric(data['frame_id']) // 10
                data['frame_id'] -= data['frame_id'].min()

                data['node_type'] = 'PEDESTRIAN'
                data['node_id'] = data['track_id'].astype(str)
                data.sort_values('frame_id', inplace=True)

                if desired_source == "eth" and data_class == "test":
                    data['pos_x'] *= 0.6
                    data['pos_y'] *= 0.6

                data['pos_x'] -= data['pos_x'].mean()
                data['pos_y'] -= data['pos_y'].mean()

                max_timesteps = data['frame_id'].max()

                scene = Scene(
                    timesteps=max_timesteps + 1,
                    dt=dt,
                    name=f'{desired_source}_{data_class}',
                    aug_func=augment if data_class == 'train' else None
                )

                for node_id in pd.unique(data['node_id']):
                    node_df = data[data['node_id'] == node_id]
                    if len(node_df) < 2:
                        continue

                    x = node_df['pos_x'].values
                    y = node_df['pos_y'].values

                    vx = derivative_of(x, scene.dt)
                    vy = derivative_of(y, scene.dt)

                    node_data = pd.DataFrame({
                        ('position', 'x'): x,
                        ('position', 'y'): y,
                        ('velocity', 'x'): vx,
                        ('velocity', 'y'): vy
                    }, columns=data_columns)

                    node = Node(
                        node_type=env.NodeType.PEDESTRIAN,
                        node_id=node_id,
                        data=node_data
                    )
                    node.first_timestep = node_df['frame_id'].iloc[0]
                    scene.nodes.append(node)

                if data_class == 'train':
                    scene.augmented = []
                    for angle in np.arange(0, 360, 15):
                        scene.augmented.append(augment_scene(scene, angle))

                scenes.append(scene)

        env.scenes = scenes

        if scenes:
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

# exit()

# =========================
# Stanford Drone Dataset
# =========================
data_columns = pd.MultiIndex.from_product(
    [['position', 'velocity'], ['x', 'y']]
)

for data_class in ['train', 'test']:
    raw_path = 'raw_data/stanford'
    out_path = 'processed_data_noise'
    data_path = os.path.join(raw_path, f'{data_class}_trajnet.pkl')

    print(f'Processing SDD {data_class}')
    df = pickle.load(open(data_path, 'rb'))

    env = Environment(
        node_type_list=['PEDESTRIAN'],
        standardization=standardization
    )

    env.attention_radius = {
        (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 3.0
    }

    scenes = []

    for scene_id, data in df.groupby('sceneId'):
        data['frame'] = pd.to_numeric(data['frame']) // 12
        data['frame'] -= data['frame'].min()

        data['node_id'] = data['trackId'].astype(str)
        data['x'] /= 50
        data['y'] /= 50

        data['x'] -= data['x'].mean()
        data['y'] -= data['y'].mean()

        max_timesteps = data['frame'].max()
        scene = Scene(
            timesteps=max_timesteps + 1,
            dt=dt,
            name=f'sdd_{data_class}',
            aug_func=augment if data_class == 'train' else None
        )

        for node_id in pd.unique(data['node_id']):
            node_df = data[data['node_id'] == node_id]
            if len(node_df) < 2:
                continue

            x = node_df['x'].values
            y = node_df['y'].values

            vx = derivative_of(x, scene.dt)
            vy = derivative_of(y, scene.dt)

            node_data = pd.DataFrame({
                ('position', 'x'): x,
                ('position', 'y'): y,
                ('velocity', 'x'): vx,
                ('velocity', 'y'): vy
            }, columns=data_columns)

            node = Node(
                node_type=env.NodeType.PEDESTRIAN,
                node_id=node_id,
                data=node_data
            )
            node.first_timestep = node_df['frame'].iloc[0]
            scene.nodes.append(node)

        if data_class == 'train':
            scene.augmented = []
            for angle in np.arange(0, 360, 15):
                scene.augmented.append(augment_scene(scene, angle))

        scenes.append(scene)

    env.scenes = scenes

    with open(os.path.join(out_path, f'sdd_{data_class}.pkl'), 'wb') as f:
        dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL)

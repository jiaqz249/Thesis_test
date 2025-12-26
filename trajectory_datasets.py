import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Build training samples from processed Environment / Scene / Node objects.
    Each sample corresponds to ONE scene window (multiple agents).
    """

    def __init__(self, env, obs_len: int, pred_len: int, attention_radius: float):
        self.env = env
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.attention_radius = attention_radius

        # index: list of (scene, start_t)
        self.index = []
        self._build_index()

    def _build_index(self):
        """
        Enumerate all valid (scene, t_start) windows.
        """
        for scene in self.env.scenes:
            T = int(scene.timesteps)
            for t in range(0, T - self.obs_len - self.pred_len + 1):
                # check at least one valid agent
                valid_nodes = []
                for node in scene.nodes:
                    t0 = node.first_timestep
                    t1 = t0 + node.data.position.x.shape[0]
                    if t >= t0 and (t + self.obs_len + self.pred_len) <= t1:
                        valid_nodes.append(node)
                if len(valid_nodes) > 0:
                    self.index.append((scene, t))

    def __len__(self):
        return len(self.index)

    # def __getitem__(self, idx):
    #     scene, t_start = self.index[idx]

    #     xs, ys, vels, poss = [], [], [], []
    #     agent_ids = []

    #     for node in scene.nodes:
    #         t0 = node.first_timestep
    #         t1 = t0 + node.data.position.x.shape[0]
    #         if t_start < t0 or (t_start + self.obs_len + self.pred_len) > t1:
    #             continue

    #         s = t_start - t0
    #         obs = node.data.iloc[s : s + self.obs_len]
    #         fut = node.data.iloc[s + self.obs_len : s + self.obs_len + self.pred_len]

    #         x = obs[[('position', 'x'), ('position', 'y')]].values.astype(np.float32)
    #         y = fut[[('position', 'x'), ('position', 'y')]].values.astype(np.float32)
    #         v = obs[[('velocity', 'x'), ('velocity', 'y')]].values.astype(np.float32)

    #         xs.append(x)
    #         ys.append(y)
    #         vels.append(v[-1])
    #         poss.append(x[-1])
    #         agent_ids.append(node)

    #     xs = np.stack(xs)          # (N, H, 2)
    #     ys = np.stack(ys)          # (N, T, 2)
    #     vels = np.stack(vels)      # (N, 2)
    #     poss = np.stack(poss)      # (N, 2)

    #     nei = self._build_neighbors(poss)

    #     return {
    #         'x': xs,
    #         'y': ys,
    #         'vel': vels,
    #         'pos': poss,
    #         'nei': nei,
    #     }
    
    def __getitem__(self, idx):
        scene, t_start = self.index[idx]

        xs, ys, vels, poss = [], [], [], []
        agent_ids = []

        for node in scene.nodes:
            t0 = int(node.first_timestep)
            T = int(node.data.position.x.shape[0])
            t1 = t0 + T

            if t_start < t0 or (t_start + self.obs_len + self.pred_len) > t1:
                continue

            s = int(t_start - t0)

            # observation
            obs_px = node.data.position.x[s : s + self.obs_len]
            obs_py = node.data.position.y[s : s + self.obs_len]
            obs_vx = node.data.velocity.x[s : s + self.obs_len]
            obs_vy = node.data.velocity.y[s : s + self.obs_len]

            # future
            fut_px = node.data.position.x[
                s + self.obs_len : s + self.obs_len + self.pred_len
            ]
            fut_py = node.data.position.y[
                s + self.obs_len : s + self.obs_len + self.pred_len
            ]

            #  4D input
            x = np.stack(
                [obs_px, obs_py, obs_vx, obs_vy],
                axis=-1
            ).astype(np.float32)        # (obs_len, 4)

            y = np.stack([fut_px, fut_py], axis=-1).astype(np.float32)      # (fut_len, 2)

            v = np.stack([obs_vx, obs_vy], axis=-1).astype(np.float32)      # (obs_len, 2)

            xs.append(x)
            ys.append(y)
            vels.append(v[-1])          # (2,)
            poss.append(x[-1, :2])      # (2,)

            agent_ids.append(node)

        xs = np.stack(xs)      # (N, obs_len, 2)
        ys = np.stack(ys)      # (N, pred_len, 2)
        vels = np.stack(vels)  # (N, 2)
        poss = np.stack(poss)  # (N, 2)

        nei = self._build_neighbors(poss)

        return {
            'x': torch.from_numpy(xs),
            'y': torch.from_numpy(ys),
            'vel': torch.from_numpy(vels),
            'pos': torch.from_numpy(poss),
            'nei': torch.from_numpy(nei),
        }


    def _build_neighbors(self, pos):
        """
        Build adjacency matrix based on distance threshold.
        pos: (N, 2)
        """
        N = pos.shape[0]
        nei = np.zeros((self.obs_len, N, N), dtype=np.int64)
        dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        mask = (dist < self.attention_radius) & (dist > 0)
        nei[:] = mask.astype(np.int64)
        return nei



def trajectory_collate(batch):
    """
    Collate multiple scenes into one batch.
    """
    xs, ys, vels, poss = [], [], [], []
    nei_lists = []
    batch_splits = []

    offset = 0
    for item in batch:
        n = item['x'].shape[0]
        xs.append(item['x'])
        ys.append(item['y'])
        vels.append(item['vel'])
        poss.append(item['pos'])
        nei_lists.append(item['nei'])
        batch_splits.append([offset, offset + n])
        offset += n

    xs = torch.cat(xs, dim=0)      # (B, H, 4)
    ys = torch.cat(ys, dim=0)      # (B, T, 2)
    vels = torch.cat(vels, dim=0)  # (B, 2)
    poss = torch.cat(poss, dim=0)  # (B, 2)

    return xs, ys, vels, poss, nei_lists, batch_splits

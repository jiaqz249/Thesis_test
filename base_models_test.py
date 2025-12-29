import pdb
from typing import Optional, Tuple, List
import argparse
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

from gca_dropin_stack import MultiLayerGCABlock
from decoder import Decoder


class GatedFusion(nn.Module):
    def __init__(self, args: argparse.Namespace, hidden_size: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_weight = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Sigmoid(),
        )

    def forward(self, a: th.Tensor, b: th.Tensor) -> th.Tensor:
        fuse = th.cat((a, b), dim=-1)
        w = self.encoder_weight(fuse)
        return a * w + b * (1 - w)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LinearPredictor(nn.Module):
    """
    Velocity-first LinearPredictor
    - Temporal backbone (Transformer + GRU) ONLY models (vx, vy)
    - (x, y) are used purely as conditioning
    """

    def __init__(self, args: argparse.Namespace, device: th.device) -> None:
        super().__init__()
        self.device = device
        self.input_len = args.obs_length
        self.output_len = args.pred_length
        self.embedding_size = args.embedding_size
        self.social_ctx_dim = args.social_ctx_dim
        self.num_samples = args.num_samples

        # ===== Patch embedding (velocity only) =====
        self.patch_mlp_vx = nn.Sequential(
            nn.Linear(3, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.patch_mlp_vy = nn.Sequential(
            nn.Linear(3, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.temporal_fusion_vel = GatedFusion(args, hidden_size=self.embedding_size)

        # ===== Positional encoding + temporal backbone =====
        self.positional_encoding = PositionalEncoding(
            d_model=self.embedding_size, dropout=0.05, max_len=32
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=args.x_encoder_head,
            dim_feedforward=self.embedding_size,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.embedding_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.enc_out_norm = nn.LayerNorm(self.embedding_size)

        # ===== Position conditioning =====
        self.pos_emb = nn.Sequential(
            nn.Linear(2, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.temporal_fusion_pos = GatedFusion(args, hidden_size=self.embedding_size)

        # ===== Social encoding =====
        self.inverted_mlp = nn.Sequential(
            nn.Linear(self.input_len, self.social_ctx_dim),
            nn.LayerNorm(self.social_ctx_dim),
            nn.ReLU(inplace=True),
        )
        self.social_channel_fusion = GatedFusion(args, hidden_size=self.social_ctx_dim)
        self.gca_dropin_stack_layer = MultiLayerGCABlock(
            edge_feat_dim=3,
            node_feat_dim=self.social_ctx_dim,
            num_heads=4,
            num_layers=1,
        )

        # ===== Decoder =====
        self.decoder = Decoder(
            enc_dim=self.embedding_size,
            social_dim=self.social_ctx_dim,
            hidden_dim=256,
            num_modes=self.num_samples,
            pred_len=self.output_len,
            num_layers=4,
            num_heads=4,
            use_self_attn=True,
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def patch_embedding(self, x: th.Tensor) -> th.Tensor:
        """
        x: (B, H, 4) = [x, y, vx, vy]
        Return: velocity-only temporal embedding (B, H-2, E)
        """
        patches = x.permute(0, 2, 1).unfold(dimension=2, size=3, step=1)
        vx = self.patch_mlp_vx(patches[:, 2])
        vy = self.patch_mlp_vy(patches[:, 3])
        vel_feat = self.temporal_fusion_vel(vx, vy)
        return vel_feat

    def inverted_embedding(self, x: th.Tensor) -> th.Tensor:
        x_inv = self.inverted_mlp(x.permute(0, 2, 1))  # (B, 4, S)
        pos_emb = self.social_channel_fusion(x_inv[:, 0], x_inv[:, 1])
        vel_emb = self.social_channel_fusion(x_inv[:, 2], x_inv[:, 3])
        return pos_emb + vel_emb

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, vel, pos, nei_lists, batch_splits):
        # ===== Temporal encoder (velocity-first) =====
        x_emb = self.patch_embedding(x)  # (B, H-2, E)
        x_emb = self.positional_encoding(x_emb)
        enc_out = self.transformer_encoder(x_emb)
        enc_out, _ = self.gru(enc_out)
        enc_out = self.enc_out_norm(enc_out)

        # ===== Position conditioning (per timestep) =====
        pos_emb = self.pos_emb(pos).unsqueeze(1).repeat(1, enc_out.size(1), 1)
        enc_out = self.temporal_fusion_pos(enc_out, pos_emb)

        # ===== Social context =====
        nei_emb = self.inverted_embedding(x)
        social_ctx = self._get_social_context(vel, pos, nei_emb, nei_lists, batch_splits)

        # ===== Decoder =====
        traj, mode_logits = self.decoder(
            enc_out=enc_out,
            social_ctx=social_ctx,
            last_obs_pos=pos,
        )

        traj = traj.transpose(0, 1)  # (K, B, T, 2)
        mode_prob = F.softmax(mode_logits, dim=-1)
        return traj, mode_logits, mode_prob

    # ------------------------------------------------------------------
    # Social context (unchanged)
    # ------------------------------------------------------------------
    def _get_social_context(
        self,
        pseudo_vel_batch: th.Tensor,
        current_pos_batch: th.Tensor,
        node_feature_batch: th.Tensor,
        nei_lists: List[np.ndarray],
        batch_splits: List[List[int]],
    ) -> th.Tensor:
        refined_feature_batch = th.zeros_like(node_feature_batch, device=self.device)
        batch_node_mask = th.full(
            (node_feature_batch.shape[0],), False, dtype=th.bool, device=self.device
        )
        graphs = []

        for batch_idx in range(len(nei_lists)):
            left, right = batch_splits[batch_idx]
            node_features = node_feature_batch[left:right]
            pseudo_vel = pseudo_vel_batch[left:right]
            ped_num = node_features.shape[0]
            nei_index = nei_lists[batch_idx][self.input_len - 1].to(self.device)

            scene_graph = dgl.graph(
                nei_index.nonzero().unbind(1), num_nodes=ped_num, device=self.device
            )

            scene_graph.ndata["h"] = node_features
            if ped_num > 1 and scene_graph.number_of_edges() > 0:
                corr = current_pos_batch[left:right].repeat(ped_num, 1, 1)
                corr_index = corr.transpose(0, 1) - corr
                dist_mat = th.norm(corr_index, dim=2)
                theta = th.atan2(pseudo_vel[:, 1], pseudo_vel[:, 0])
                bearing = th.atan2(corr_index[:, :, 1], corr_index[:, :, 0])
                bearing_rel = bearing - theta.view(1, -1)
                heading_diff = theta.view(1, -1) - theta.view(-1, 1)

                def wrap(x):
                    return (x + th.pi) % (2 * th.pi) - th.pi

                edge_feat = th.stack(
                    (
                        dist_mat.reshape(-1),
                        wrap(bearing_rel).reshape(-1),
                        wrap(heading_diff).reshape(-1),
                    ),
                    dim=-1,
                )
                mask = nei_index.view(-1) > 0
                scene_graph.edata["feat"] = edge_feat[mask]
                batch_node_mask[left:right] = nei_index.sum(dim=1) > 0

            graphs.append(scene_graph)

        graph_batch = dgl.batch(graphs)
        refined_feature_batch[batch_node_mask] = self.gca_dropin_stack_layer(
            graph_batch
        )[batch_node_mask]
        return refined_feature_batch

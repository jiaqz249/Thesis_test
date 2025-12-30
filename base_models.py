import pdb
from typing import Optional, Tuple, List
import argparse
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
# from gca_models import GraphConstrainedAttentionLayer
# from gca_dropin import GraphConstrainedAttentionLayer
from gca_dropin_stack import MultiLayerGCABlock
from decoder import Decoder


class GatedFusion(nn.Module):

    def __init__(self, args: argparse.Namespace, hidden_size: Optional[int] = None) -> None:
        super(GatedFusion, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_weight = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Sigmoid()
        )

    def forward(self, past_feat: th.Tensor, dest_feat: th.Tensor) -> th.Tensor:
        fuse = th.cat((past_feat, dest_feat), dim=-1)
        weight = self.encoder_weight(fuse)
        fused = past_feat * weight + dest_feat * (1 - weight)
        return fused



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LinearPredictor(nn.Module):
    '''
    main func of the pmm-net
    '''
    def __init__(self, args: argparse.Namespace, device: th.device) -> None:
        super(LinearPredictor, self).__init__()
        self.device = device
        self.input_dim = args.input_size
        self.output_dim = args.output_size
        self.input_len = args.obs_length  # type: int
        self.output_len = args.pred_length  # type: int
        self.embedding_size = args.embedding_size  # E
        self.num_tokens = self.input_len - 2  # self.input_len - 2
        self.social_ctx_dim = args.social_ctx_dim  # S
        self.num_samples = args.num_samples  # K
        self.positional_encoding = PositionalEncoding(d_model=self.embedding_size, dropout=0.05, max_len=24)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=args.x_encoder_head,
            dim_feedforward=self.embedding_size,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.patch_mlp_x = nn.Sequential(
            nn.Linear(3, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.patch_mlp_y = nn.Sequential(
            nn.Linear(3, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(inplace=True),
        )
        self.temporal_channel_fusion = GatedFusion(args, hidden_size=self.embedding_size)
        self.social_channel_fusion = GatedFusion(args, hidden_size=self.social_ctx_dim)
        self.gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.embedding_size,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.1,
            bidirectional=False,
        )
        self.inverted_mlp = nn.Sequential(
            nn.Linear(self.input_len, self.social_ctx_dim),
            nn.LayerNorm(self.social_ctx_dim),
            nn.ReLU(inplace=True),
        )
        # self.gca_layer = GraphConstrainedAttentionLayer(edge_feat_dim=3, node_feat_dim=self.social_ctx_dim, num_heads=4)
        # self.gca_dropin_layer = GraphConstrainedAttentionLayer(edge_feat_dim=3,
        #                                                 node_feat_dim=self.social_ctx_dim,
        #                                                 num_heads=4,  # 256 % 4 == 0
        # )
        self.gca_dropin_stack_layer = MultiLayerGCABlock(edge_feat_dim=3,
                                            node_feat_dim=self.social_ctx_dim,
                                            num_heads=4,
                                            num_layers=1,
        )
        self.temporal_fusion_pos = GatedFusion(args, hidden_size=self.embedding_size)
        self.temporal_fusion_vel = GatedFusion(args, hidden_size=self.embedding_size)
        self.temporal_fusion_all = GatedFusion(args, hidden_size=self.embedding_size)
        self.decoder = Decoder(enc_dim=self.embedding_size,
                               social_dim=self.social_ctx_dim,
                               hidden_dim=128,
                               num_modes=self.num_samples,
                               pred_len=self.output_len,
                               num_layers=4,
                               num_heads=4,
                               use_self_attn=True,
                               )
        self.enc_out_norm = nn.LayerNorm(self.embedding_size)
        self.prob_dropout = nn.Dropout(p=0.1)

    def patch_embedding(self, x: th.Tensor) -> th.Tensor:
        # x: (B, H, 4)
        patches = x.permute(0, 2, 1).unfold(dimension=2, size=3, step=1)
        # (B, 4, H-2, 3)

        patch_emb_x  = self.patch_mlp_x(patches[:, 0])  # (B, H-2, E)
        patch_emb_y  = self.patch_mlp_y(patches[:, 1])
        patch_emb_vx = self.patch_mlp_x(patches[:, 2])
        patch_emb_vy = self.patch_mlp_y(patches[:, 3])

        pos_feat = self.temporal_fusion_pos(patch_emb_x, patch_emb_y)
        vel_feat = self.temporal_fusion_vel(patch_emb_vx, patch_emb_vy)

        fused_feat = self.temporal_fusion_all(pos_feat, vel_feat)
        return fused_feat


    def inverted_embedding(self, x: th.Tensor) -> th.Tensor:
        # x: (B, H, 4)
        x_inv_emb = self.inverted_mlp(x.permute(0, 2, 1))  # (B, 4, S)

        pos_emb = self.social_channel_fusion(
            x_inv_emb[:, 0],  # x
            x_inv_emb[:, 1],  # y
        )

        vel_emb = self.social_channel_fusion(
            x_inv_emb[:, 2],  # vx
            x_inv_emb[:, 3],  # vy
        )

        nei_emb = pos_emb + vel_emb
        return nei_emb  # (B, S)


    def forward(self, x: th.Tensor, vel, pos, nei_lists, batch_splits):

        # encoder
        x_emb = self.patch_embedding(x)                 # (B, H - 2, E)
        nei_emb = self.inverted_embedding(x)            # (B, S)
        social_ctx = self._get_social_context(
            vel, pos, nei_emb, nei_lists, batch_splits
        )                                                # (B, S)

        x_emb = self.positional_encoding(
            x_emb.permute(1, 0, 2)
        ).permute(1, 0, 2)

        enc_out = self.transformer_encoder(x_emb)
        enc_out, _ = self.gru(enc_out)
        enc_out = self.enc_out_norm(enc_out)

        # >>> 新增：last observed position
        last_obs_pos = pos                   # (B, 2)

        # decoder
        traj, mode_logits = self.decoder(
            enc_out=enc_out,
            social_ctx=social_ctx,
            last_obs_pos=last_obs_pos
        )

        # traj: (B, K, T, 2) → (K, B, T, 2)
        traj = traj.transpose(0, 1)

        mode_prob = F.softmax(mode_logits, dim=-1)

        return traj, mode_logits, mode_prob



    def _get_social_context(
        self,
        pseudo_vel_batch: th.Tensor,
        current_pos_batch: th.Tensor,
        node_feature_batch: th.Tensor,
        nei_lists: List[np.ndarray],
        batch_splits: List[List[int]],
    ) -> th.Tensor:
        # nei_lists: (B, H, N, N), current_pos_batch (B, 2)
        refined_feature_batch = th.zeros_like(node_feature_batch, device=self.device)
        batch_node_mask = th.full((node_feature_batch.shape[0],), False, dtype=th.bool, device=self.device)
        graphs = []

        for batch_idx in range(len(nei_lists)):
            # ids for select agents in the same scene
            left, right = batch_splits[batch_idx][0], batch_splits[batch_idx][1]
            node_features = node_feature_batch[left: right]  # (N, S)
            pseudo_vel = pseudo_vel_batch[left: right, :]  # velocity of the agent
            ped_num = node_features.shape[0]  # num_nodes in current scene
            nei_index = nei_lists[batch_idx][self.input_len - 1].to(self.device)    # (N, N)
            assert node_features.shape[0] == nei_index.shape[0], \
    f"Mismatch: node_features {node_features.shape}, nei_index {nei_index.shape}"
            # build graph based on adjacency matrix
            scene_graph = dgl.graph(nei_index.nonzero().unbind(1), num_nodes=ped_num, device=self.device)

            if ped_num != 1:
                
                corr = current_pos_batch[left: right].repeat(ped_num, 1, 1)  # (N, N, 2)
                corr_index = corr.transpose(0, 1) - corr                     # (N, N, 2)

                pseudo_vel_norm = th.norm(pseudo_vel, dim=1, keepdim=True)    # (N, 1)
                theta = th.atan2(pseudo_vel[:, 1], pseudo_vel[:, 0])         # (N,)

                # zero-velocity handling
                zero_velocity_mask = pseudo_vel_norm.squeeze() == 0
                if zero_velocity_mask.any():
                    theta = theta.clone()
                    theta[zero_velocity_mask] = 0.0

                dist_mat = th.norm(corr_index, dim=2)                         # (N, N)

                bearing = th.atan2(corr_index[:, :, 1], corr_index[:, :, 0])  # (N, N)
                bearing_rel = bearing - theta.view(1, -1)                     # (N, N)

                heading_diff = theta.view(1, -1) - theta.view(-1, 1)          # (N, N)

                def wrap_angle(x: th.Tensor) -> th.Tensor:
                    return (x + th.pi) % (2 * th.pi) - th.pi

                bearing_rel = wrap_angle(bearing_rel)
                heading_diff = wrap_angle(heading_diff)

                edge_feature_mat = th.stack(
                    (
                        dist_mat.reshape(-1),
                        bearing_rel.reshape(-1),
                        heading_diff.reshape(-1),
                    ),
                    dim=-1
                )  # (N * N, 3)

                edge_mask = nei_index.view(ped_num * ped_num) > 0

                node_mask = th.sum(nei_index, dim=1) > 0
                batch_node_mask[left: right] = node_mask

                scene_graph.ndata["h"] = node_features
                if scene_graph.number_of_edges() > 0:
                    scene_graph.edata["feat"] = edge_feature_mat[edge_mask]   
                
            else:  # one node, no edge
                scene_graph.ndata["h"] = node_features

            graphs.append(scene_graph)

        graph_batch = dgl.batch(graphs)
        try:
            refined_feature_batch[batch_node_mask] = self.gca_dropin_stack_layer(graph_batch)[batch_node_mask]
            # h_new = self.gca_layer(graph_batch)
            # graph_batch.ndata["h"] = h_new
            # refined_feature_batch[batch_node_mask] = h_new[batch_node_mask]
        except IndexError:
            pdb.set_trace()
        return refined_feature_batch
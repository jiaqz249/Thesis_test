import math
import dgl
import torch as th
from torch import nn
import torch.nn.functional as F


class GraphConstrainedAttentionLayer(nn.Module):
    """
    Single Graph-Constrained Attention (GCA) layer.
    No residual inside. Residual is handled by the wrapper.
    """

    def __init__(self, edge_feat_dim: int, node_feat_dim: int, num_heads: int):
        super().__init__()
        assert node_feat_dim % num_heads == 0

        self.num_heads = num_heads
        self.node_feat_dim = node_feat_dim
        self.head_dim = node_feat_dim // num_heads

        # Q K V projections
        self.W_q = nn.Linear(node_feat_dim, node_feat_dim, bias=False)
        self.W_k = nn.Linear(node_feat_dim, node_feat_dim, bias=False)
        self.W_v = nn.Linear(node_feat_dim, node_feat_dim, bias=False)

        # edge -> per-head attention bias
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, node_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_feat_dim, num_heads),
        )

        self.out_proj = nn.Linear(node_feat_dim, node_feat_dim)

    # ---------- message ----------
    def message_func(self, edges):
        # Q_i · K_j
        attn = (edges.dst["Q"] * edges.src["K"]).sum(-1)
        attn = attn / math.sqrt(self.head_dim)

        # add edge bias
        attn = attn + self.edge_mlp(edges.data["feat"])

        return {
            "score": attn,       # (E, H)
            "V": edges.src["V"], # (E, H, D)
        }

    # ---------- reduce ----------
    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["score"], dim=1)   # over neighbors
        alpha = alpha.unsqueeze(-1)                        # (N, deg, H, 1)

        out = th.sum(alpha * nodes.mailbox["V"], dim=1)    # (N, H, D)
        out = out.reshape(out.shape[0], -1)                # (N, S)

        return {"h_new": out}

    # ---------- forward ----------
    def forward(self, g: dgl.DGLGraph, h: th.Tensor) -> th.Tensor:
        """
        h: (N, S)
        return: (N, S)
        """
        if g.number_of_edges() == 0:
            return h

        Q = self.W_q(h).view(-1, self.num_heads, self.head_dim)
        K = self.W_k(h).view(-1, self.num_heads, self.head_dim)
        V = self.W_v(h).view(-1, self.num_heads, self.head_dim)

        g.ndata["Q"] = Q
        g.ndata["K"] = K
        g.ndata["V"] = V

        g.update_all(self.message_func, self.reduce_func)

        return self.out_proj(g.ndata["h_new"])


class MultiLayerGCABlock(nn.Module):
    """
    Drop-in replacement for EdgeGraphConvLayer.

    Interface:
        input : g.ndata["h"] (N, S), g.edata["feat"] (E, edge_feat_dim)
        output: Tensor (N, S)

    Usage:
        self.gnn_layer = MultiLayerGCABlock(...)
    """

    def __init__(
        self,
        edge_feat_dim: int,
        node_feat_dim: int,
        num_heads: int,
        num_layers: int = 3,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            GraphConstrainedAttentionLayer(
                edge_feat_dim=edge_feat_dim,
                node_feat_dim=node_feat_dim,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(node_feat_dim)
            for _ in range(num_layers)
        ])

        self.activation = nn.PReLU()

    def forward(self, g: dgl.DGLGraph):
        """
        EXACT SAME forward signature as EdgeGraphConvLayer
        """
        h = g.ndata["h"]

        for gca, norm in zip(self.layers, self.norms):
            h = h + self.activation(norm(gca(g, h)))
        
        g.ndata["h"] = h

        return g.ndata["h"]

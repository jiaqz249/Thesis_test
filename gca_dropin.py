import math
import dgl
import torch as th
from torch import nn
import torch.nn.functional as F


class GraphConstrainedAttentionLayer(nn.Module):
    """
    Drop-in replacement for EdgeGraphConvLayer.

    Behavior:
    - reads g.ndata["h"], g.edata["feat"]
    - updates g.ndata["h"] in-place
    - returns g.ndata["h"]
    """

    def __init__(
        self,
        edge_feat_dim: int,
        node_feat_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        assert node_feat_dim % num_heads == 0

        self.num_heads = num_heads
        self.node_feat_dim = node_feat_dim
        self.head_dim = node_feat_dim // num_heads

        # Q K V projections
        self.W_q = nn.Linear(node_feat_dim, node_feat_dim, bias=False)
        self.W_k = nn.Linear(node_feat_dim, node_feat_dim, bias=False)
        self.W_v = nn.Linear(node_feat_dim, node_feat_dim, bias=False)

        # edge → attention bias (per head)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, node_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_feat_dim, num_heads),
        )

        # output
        self.out_proj = nn.Linear(node_feat_dim, node_feat_dim)
        self.output_activation = nn.PReLU()

    # ---------- message ----------
    def message_func(self, edges):
        # edges.dst["Q"], edges.src["K"]: [E, H, D]
        attn_score = (edges.dst["Q"] * edges.src["K"]).sum(-1)
        attn_score = attn_score / math.sqrt(self.head_dim)

        # edge bias
        edge_bias = self.edge_mlp(edges.data["feat"])  # [E, H]
        attn_score = attn_score + edge_bias

        return {
            "score": attn_score,    # [E, H]
            "V": edges.src["V"],    # [E, H, D]
        }

    # ---------- reduce ----------
    def reduce_func(self, nodes):
        # nodes.mailbox["score"]: [N, deg, H]
        # nodes.mailbox["V"]:     [N, deg, H, D]
        alpha = F.softmax(nodes.mailbox["score"], dim=1)
        alpha = alpha.unsqueeze(-1)                    # [N, deg, H, 1]

        out = th.sum(alpha * nodes.mailbox["V"], dim=1)  # [N, H, D]
        out = out.reshape(out.shape[0], -1)              # [N, S]

        return {"h_new": out}

    # ---------- forward ----------
    def forward(self, g: dgl.DGLGraph):
        # EdgeGraphConvLayer compatibility
        if g.number_of_edges() == 0:
            return g.ndata["h"]

        h = g.ndata["h"]  # [N, S]

        # Q K V
        Q = self.W_q(h).view(-1, self.num_heads, self.head_dim)
        K = self.W_k(h).view(-1, self.num_heads, self.head_dim)
        V = self.W_v(h).view(-1, self.num_heads, self.head_dim)

        g.ndata["Q"] = Q
        g.ndata["K"] = K
        g.ndata["V"] = V

        g.update_all(self.message_func, self.reduce_func)

        # residual + activation
        h_out = self.output_activation(
            self.out_proj(g.ndata["h_new"]) + h
        )

        #  drop-in 核心：写回 graph
        g.ndata["h"] = h_out

        return g.ndata["h"]

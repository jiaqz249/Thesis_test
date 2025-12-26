import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention + FFN block.
    Self-attention is OPTIONAL and OFF by default.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_self_attn: bool = False,  # interface preserved
    ):
        super().__init__()

        self.use_self_attn = use_self_attn

        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                batch_first=True,
            )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.norm_q_sa = nn.LayerNorm(dim)
        self.norm_q_ca = nn.LayerNorm(dim)
        self.norm_m = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, q: th.Tensor, memory: th.Tensor) -> th.Tensor:
        """
        Args:
            q:      (B, K, D)   query
            memory: (B, M, D)   key / value
        """

        # ---- optional self-attention (NOT used by default) ----
        if self.use_self_attn:
            q_sa, _ = self.self_attn(
                self.norm_q_sa(q),
                self.norm_q_sa(q),
                self.norm_q_sa(q),
            )
            q = q + q_sa

        # ---- cross-attention ----
        q_ca, _ = self.cross_attn(
            self.norm_q_ca(q),
            self.norm_m(memory),
            self.norm_m(memory),
        )
        q = q + q_ca

        # ---- FFN ----
        q = q + self.mlp(q)
        q = self.norm_ffn(q)

        return q


class Decoder(nn.Module):
    """
    QCNet-style decoder (minimal invasive version):

    - proposal trajectory is built iteratively
    - refinement is single-shot
    """

    def __init__(
        self,
        enc_dim: int,
        social_dim: int,
        hidden_dim: int,
        num_modes: int,
        pred_len: int,
        num_layers: int,
        num_heads: int = 4,
        use_self_attn: bool = False,
    ):
        super().__init__()

        self.num_modes = num_modes
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim

        # ---------- memory projection ----------
        self.enc_proj = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.social_proj = nn.Sequential(
            nn.Linear(social_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ---------- learnable mode queries ----------
        self.query_embed = nn.Parameter(
            th.randn(1, num_modes, hidden_dim)
        )

        # ---------- recursive proposal layers ----------
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                use_self_attn=use_self_attn,
            )
            for _ in range(num_layers)
        ])

        # ---------- proposal delta head ----------
        self.proposal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pred_len * 2),
        )

        # ---------- trajectory embedding ----------
        self.traj_embed = nn.Sequential(
            nn.Linear(pred_len * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ---------- refinement (single-shot) ----------
        self.refine_block = CrossAttentionBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            use_self_attn=False,
        )

        # ---------- probability ----------
        self.prob_head = nn.Linear(hidden_dim, 1)

        # ---------- final output ----------
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pred_len * 2),
        )
        self.delta_dropout = nn.Dropout(p=0.1)
        self.delta_scale = 0.5

    def forward(
        self,
        enc_out: th.Tensor,
        social_ctx: th.Tensor,
    ):
        """
        Returns:
            traj_final: (B, K, T, 2)
            mode_logits: (B, K)
        """

        B = enc_out.size(0)
        device = enc_out.device

        # ---------- build memory ----------
        enc_mem = self.enc_proj(enc_out)                    # (B, H-2, D)
        soc_mem = self.social_proj(social_ctx).unsqueeze(1) # (B, 1, D)
        memory = th.cat([enc_mem, soc_mem], dim=1)          # (B, H-1, D)

        # ---------- init queries ----------
        q = self.query_embed.expand(B, -1, -1)              # (B, K, D)

        # ---------- init proposal trajectory ----------
        traj_prop = th.zeros(
            B, self.num_modes, self.pred_len, 2,
            device=device
        )

        # ---------- iterative proposal decoding ----------
        for layer in self.layers:
            # cross-attend memory
            q = layer(q, memory)

            # predict delta trajectory
            delta = self.proposal_head(q)
            delta = self.delta_dropout(delta)
            delta = self.delta_scale * delta
            delta = delta.view(
                B, self.num_modes, self.pred_len, 2
            )

            # update proposal
            traj_prop = traj_prop + delta

            # re-embed trajectory for next step
            q = self.traj_embed(
                traj_prop.view(B, self.num_modes, -1)
            )

        # ---------- single-shot refinement ----------
        q_ref = self.refine_block(q, memory)

        mode_logits = self.prob_head(q_ref).squeeze(-1)

        traj_final = self.out_head(q_ref)
        traj_final = traj_final.view(
            B, self.num_modes, self.pred_len, 2
        )

        return traj_final, mode_logits


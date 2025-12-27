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

        # ---------- cross-attn layers ----------
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                use_self_attn=use_self_attn,
            )
            for _ in range(num_layers)
        ])

        # ---------- endpoint head ----------
        self.endpoint_head = nn.Linear(hidden_dim, 2)

        # ---------- residual head ----------
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        # ---------- refinement ----------
        self.refine_block = CrossAttentionBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            use_self_attn=False,
        )

        # ---------- probability ----------
        self.prob_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        enc_out: th.Tensor,
        social_ctx: th.Tensor,
        last_obs_pos: th.Tensor = None,  # (B, 2)
    ):
        """
        Args:
            enc_out: (B, H, enc_dim)
            social_ctx: (B, social_dim)
            last_obs_pos: (B, 2)
        Returns:
            traj_final: (B, K, T, 2)
            mode_logits: (B, K)
        """

        B = enc_out.size(0)
        device = enc_out.device

        # ---------- memory ----------
        enc_mem = self.enc_proj(enc_out)              # (B, H, D)
        soc_mem = self.social_proj(social_ctx).unsqueeze(1)  # (B, 1, D)
        memory = th.cat([enc_mem, soc_mem], dim=1)   # (B, H+1, D)

        # ---------- init queries ----------
        mode_latent = th.randn(B, self.num_modes, self.hidden_dim, device=device)
        q = self.query_embed.expand(B, -1, -1) + mode_latent  # (B, K, D)

        # ---------- cross-attention ----------
        for layer in self.layers:
            q = layer(q, memory)  # (B, K, D)

        # ---------- endpoint prediction ----------
        endpoint = self.endpoint_head(q)  # (B, K, 2)

        # ---------- baseline trajectory ----------
        if last_obs_pos is None:
            start = th.zeros(B, 1, 2, device=device)  # (B, 1, 2)
        else:
            start = last_obs_pos.unsqueeze(1)        # (B, 1, 2)

        # 线性插值生成粗轨迹
        t = th.linspace(1, self.pred_len, self.pred_len, device=device).view(1, 1, -1, 1) / self.pred_len
        baseline = start[:, :, None, :] + t * (endpoint[:, :, None, :] - start[:, :, None, :])
        # baseline shape: (B, K, T, 2)

        # ---------- residual ----------
        # 扩展 q 和 endpoint 到每个时间步
        q_expand = q.unsqueeze(2).expand(-1, -1, self.pred_len, -1)          # (B, K, T, D)
        endpoint_expand = endpoint.unsqueeze(2).expand(-1, -1, self.pred_len, -1)  # (B, K, T, 2)

        residual = self.residual_head(th.cat([q_expand, endpoint_expand], dim=-1))  # (B, K, T, 2)
        traj = baseline + residual  # (B, K, T, 2)

        # ---------- refinement ----------
        q_ref = self.refine_block(q, memory)  # (B, K, D)
        mode_logits = self.prob_head(q_ref).squeeze(-1)  # (B, K)

        return traj, mode_logits



import dill
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from types import SimpleNamespace

from trajectory_datasets import TrajectoryDataset, trajectory_collate
from base_models import LinearPredictor

# load data
env = dill.load(open('processed_data_noise/eth_train.pkl', 'rb'))

dataset = TrajectoryDataset(
    env,
    obs_len=8,
    pred_len=12,
    attention_radius=3.0
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=trajectory_collate
)

args = SimpleNamespace(
    input_size=4,
    output_size=2,
    obs_length=8,
    pred_length=12,

    embedding_size=64,
    hidden_size=1024,
    social_ctx_dim=256,
    num_samples=20,

    x_encoder_head=4,
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LinearPredictor(args, device).to(device)

with torch.no_grad():
    for batch in loader:
        x, y, vel, pos, nei_lists, batch_splits = batch
        x = x.to(device)
        vel = vel.to(device)
        pos = pos.to(device)
        print("dicos")
        out, _, _ = model(x, vel, pos, nei_lists, batch_splits)
        print(out.shape)  # (K, B, T, 2)
        break


# def min_ade_loss(pred, gt):
#     """
#     pred: (K, B, T, 2)
#     gt:   (B, T, 2)
#     """
#     gt = gt.unsqueeze(0)               # (1, B, T, 2)
#     l2 = torch.norm(pred - gt, dim=-1) # (K, B, T)
#     ade = l2.mean(dim=-1)              # (K, B)
#     min_ade, _ = ade.min(dim=0)        # (B,)
#     return min_ade.mean()

def best_of_k_ade(pred, gt):
    """
    Best-of-K ADE loss (min-over-modes).

    Args:
        pred: (K, B, T, 2)
        gt:   (B, T, 2)

    Returns:
        loss_reg: scalar
        best_k:   (B,) index of best mode
    """
    K, B, T, _ = pred.shape

    gt_exp = gt.unsqueeze(0)                       # (1, B, T, 2)
    l2 = torch.norm(pred - gt_exp, dim=-1)         # (K, B, T)
    ade = l2.mean(dim=-1)                          # (K, B)

    min_ade, best_k = ade.min(dim=0)               # (B,)
    loss_reg = min_ade.mean()

    return loss_reg, best_k


def mode_classification_loss(mode_logits, best_k):
    """
    Mode classification loss.
    Encourage the predicted mode probability to match the best trajectory.

    Args:
        mode_logits: (B, K) raw logits from decoder
        best_k:      (B,)   index of best mode (from min-ADE)

    Returns:
        loss_cls: scalar
    """
    return F.cross_entropy(mode_logits, best_k)


def trajectory_diversity_loss(pred, max_dist=10.0):
    """
    Trajectory-level diversity loss with distance cap.
    """
    K, B, T, _ = pred.shape

    diff = pred.unsqueeze(1) - pred.unsqueeze(0)     # (K, K, B, T, 2)
    dist = torch.norm(diff, dim=-1).mean(dim=(2, 3)) # (K, K)

    # ---- critical fix: cap distance ----
    dist = torch.clamp(dist, max=max_dist)

    mask = 1.0 - torch.eye(K, device=pred.device)
    loss_div = -(dist * mask).sum() / (K * (K - 1))

    return loss_div



# def smoothness_loss(pred, best_k):
#     """
#     Velocity smoothness loss (optional).
#     Only applied to best mode.

#     Args:
#         pred:   (K, B, T, 2)
#         best_k:(B,)

#     Returns:
#         loss_smooth: scalar
#     """
#     K, B, T, _ = pred.shape

#     # select best trajectories
#     best_traj = pred[best_k, torch.arange(B)]      # (B, T, 2)

#     vel = best_traj[:, 1:] - best_traj[:, :-1]     # (B, T-1, 2)
#     acc = vel[:, 1:] - vel[:, :-1]                  # (B, T-2, 2)

#     return acc.norm(dim=-1).mean()


def trajectory_loss(
    pred,
    gt,
    mode_logits,
    lambda_div=0.1,
    lambda_cls=0.0,
):
    """
    Full loss for multimodal trajectory prediction.

    Args:
        pred: (K, B, T, 2)
        gt:   (B, T, 2)

    Returns:
        total_loss: scalar
        loss_dict:  dict of components (for logging)
    """

    # -------- best-of-K regression --------
    loss_reg, best_k = best_of_k_ade(pred, gt)

    # -------- diversity loss --------
    loss_div = trajectory_diversity_loss(pred)

    # ---------- mode classification ----------
    loss_cls = mode_classification_loss(mode_logits, best_k)
    total_loss = (
        loss_reg
        + lambda_div * loss_div
        + lambda_cls * loss_cls
    )

    loss_dict = {
        "loss_total": total_loss.item(),
        "loss_reg": loss_reg.item(),
        "loss_div": loss_div.item(),
        "loss_cls": loss_cls.item(),
    }

    return total_loss, loss_dict


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(64):
    model.train()
    total_loss = 0

    for x, y, vel, pos, nei_lists, batch_splits in loader:
        x = x.to(device)
        y = y.to(device)
        vel = vel.to(device)
        pos = pos.to(device)
        # print(x.shape)  # 必须是 (B, obs_len, 4)

        pred, mode_logits, _ = model(x, vel, pos, nei_lists, batch_splits)

        if epoch < 10:
            lambda_cls = 0.0
            lambda_div = 0.0
        elif epoch < 30:
            lambda_cls = 0.3
            lambda_div = 0.05
        else:
            lambda_cls = 0.5
            lambda_div = 0.1

        loss, loss_dict = trajectory_loss(pred, y,
                                          mode_logits,
                                          lambda_div,
                                          lambda_cls,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    # print("dicos")
    print(f"Epoch {epoch}: {total_loss / len(loader):.4f}")

import os
import dill
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trajectory_datasets import TrajectoryDataset, trajectory_collate
# from base_models import LinearPredictor
from base_models_test import LinearPredictor
from losses import trajectory_loss

# =========================
# util
# =========================
def get_lambda_cls(epoch):
    if epoch < 20:
        return 0.0
    elif epoch < 60:
        return 0.4 * (epoch - 30) / 30
    else:
        return 0.4


# =========================
# metrics
# =========================
def min_ade(pred, gt):
    """
    pred: (K, B, T, 2)
    gt:   (B, T, 2)
    """
    K, B, T, _ = pred.shape
    gt = gt.unsqueeze(0).expand(K, -1, -1, -1)
    ade = torch.norm(pred - gt, dim=-1).mean(dim=-1)  # (K, B)
    return ade.min(dim=0)[0].mean()


def min_fde(pred, gt):
    """
    pred: (K, B, T, 2)
    gt:   (B, T, 2)
    """
    K, B, T, _ = pred.shape
    gt_last = gt[:, -1].unsqueeze(0).expand(K, -1, -1)
    fde = torch.norm(pred[:, :, -1] - gt_last, dim=-1)  # (K, B)
    return fde.min(dim=0)[0].mean()


# =========================
# validation
# =========================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    minade_sum, minfde_sum = 0.0, 0.0
    top1ade_sum, top1fde_sum = 0.0, 0.0
    cnt = 0

    for x, y, vel, pos, nei_lists, batch_splits in loader:
        x = x.to(device)
        y = y.to(device)
        vel = vel.to(device)
        pos = pos.to(device)

        traj, mode_logits, mode_prob = model(
            x, vel, pos, nei_lists, batch_splits
        )

        minade_sum += min_ade(traj, y).item()
        minfde_sum += min_fde(traj, y).item()

        top1ade_sum += top1_ade(traj, y, mode_prob).item()
        top1fde_sum += top1_fde(traj, y, mode_prob).item()

        cnt += 1

    return (
        minade_sum / cnt,
        minfde_sum / cnt,
        top1ade_sum / cnt,
        top1fde_sum / cnt,
    )


def top1_ade(pred, gt, mode_prob):
    """
    pred: (K, B, T, 2)
    gt:   (B, T, 2)
    mode_prob: (B, K)
    """
    B = gt.shape[0]
    top1_idx = mode_prob.argmax(dim=1)  # (B,)

    pred_top1 = pred[
        top1_idx,
        torch.arange(B, device=gt.device)
    ]  # (B, T, 2)

    l2 = torch.norm(pred_top1 - gt, dim=-1)  # (B, T)
    return l2.mean(dim=1).mean()


def top1_fde(pred, gt, mode_prob):
    B = gt.shape[0]
    top1_idx = mode_prob.argmax(dim=1)

    pred_top1 = pred[
        top1_idx,
        torch.arange(B, device=gt.device)
    ]

    return torch.norm(
        pred_top1[:, -1] - gt[:, -1], dim=-1
    ).mean()


# =========================
# train
# =========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- load env --------
    train_env = dill.load(open(args.train_pkl, "rb"))
    val_env   = dill.load(open(args.val_pkl, "rb"))

    train_set = TrajectoryDataset(
        train_env, args.obs_length, args.pred_length, args.attention_radius
    )
    val_set = TrajectoryDataset(
        val_env, args.obs_length, args.pred_length, args.attention_radius
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trajectory_collate,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=trajectory_collate,
        num_workers=0,
    )

    # -------- model --------
    model = LinearPredictor(args, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_ade = 1e9
    os.makedirs(args.ckpt_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for x, y, vel, pos, nei_lists, batch_splits in train_loader:
            x = x.to(device)
            y = y.to(device)
            vel = vel.to(device)
            pos = pos.to(device)

            pred, mode_logits, _ = model(x, vel, pos, nei_lists, batch_splits)

            lambda_cls = get_lambda_cls(epoch)
            lambda_div = 0.0
            lambda_ent = 0.0
            
            loss = trajectory_loss(
                pred, y, mode_logits,
                lambda_div=lambda_div,
                lambda_cls=lambda_cls,
                lambda_ent=lambda_ent
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ===== validation =====
        val_minade, val_minfde, val_top1ade, val_top1fde = evaluate(
            model, val_loader, device
        )

        score = val_minade + 0.3 * val_top1ade


        if epoch > 20 and score < best_val_ade:
            best_val_ade = score
            torch.save(
                model.state_dict(),
                os.path.join(args.ckpt_dir, "model_best_hotel_6.pth")
            )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={total_loss / len(train_loader):.4f} | "
            f"minADE={val_minade:.4f} "
            f"top1ADE={val_top1ade:.4f}"
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument("--train_pkl", type=str, required=True)
    # parser.add_argument("--val_pkl", type=str, required=True)
    parser.add_argument("--train_pkl", type=str, default="processed_data_noise/hotel_train.pkl")
    parser.add_argument("--val_pkl", type=str, default="processed_data_noise/hotel_val.pkl")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")

    parser.add_argument("--obs_length", type=int, default=8)
    parser.add_argument("--pred_length", type=int, default=12)
    parser.add_argument("--attention_radius", type=float, default=3.0)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-4)

    # model args
    parser.add_argument("--input_size", type=int, default=4)
    parser.add_argument("--output_size", type=int, default=2)
    parser.add_argument("--x_encoder_head", type=int, default=4)
    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--social_ctx_dim", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=20)

    args = parser.parse_args()
    main(args)

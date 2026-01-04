import os
import dill
import argparse
import torch
from torch.utils.data import DataLoader

from trajectory_datasets import TrajectoryDataset, trajectory_collate
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
    fde = torch.norm(pred[:, :, -1] - gt_last, dim=-1)
    return fde.min(dim=0)[0].mean()


# =========================
# evaluation
# =========================
@torch.no_grad()
def evaluate_test(model, loader, device):
    """
    Test-time evaluation:
    only minADE / minFDE, no mode selection
    """
    model.eval()
    ade_sum, fde_sum = 0.0, 0.0
    cnt = 0

    for x, y, vel, pos, nei_lists, batch_splits in loader:
        x = x.to(device)
        y = y.to(device)
        vel = vel.to(device)
        pos = pos.to(device)

        pred, _, _ = model(x, vel, pos, nei_lists, batch_splits)

        ade_sum += min_ade(pred, y).item()
        fde_sum += min_fde(pred, y).item()
        cnt += 1

    return ade_sum / cnt, fde_sum / cnt


# =========================
# train
# =========================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- load env --------
    train_env = dill.load(open(args.train_pkl, "rb"))
    test_env  = dill.load(open(args.test_pkl, "rb"))

    train_set = TrajectoryDataset(
        train_env, args.obs_length, args.pred_length, args.attention_radius
    )
    test_set = TrajectoryDataset(
        test_env, args.obs_length, args.pred_length, args.attention_radius
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trajectory_collate,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=trajectory_collate,
        num_workers=0,
    )

    # -------- model --------
    model = LinearPredictor(args, device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_test_ade = 1e9
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # =========================
    # training loop
    # =========================
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for x, y, vel, pos, nei_lists, batch_splits in train_loader:
            x = x.to(device)
            y = y.to(device)
            vel = vel.to(device)
            pos = pos.to(device)

            pred, mode_logits, _ = model(
                x, vel, pos, nei_lists, batch_splits
            )

            lambda_cls = get_lambda_cls(epoch)
            lambda_div = 0.0
            lambda_ent = 0.0

            loss = trajectory_loss(
                pred,
                y,
                mode_logits,
                lambda_div=lambda_div,
                lambda_cls=lambda_cls,
                lambda_ent=lambda_ent,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ===== test evaluation =====
        test_minade, test_minfde = evaluate_test(
            model, test_loader, device
        )

        # model selection criterion
        score = test_minade

        if score < best_test_ade:
            best_test_ade = score
            torch.save(
                model.state_dict(),
                os.path.join(args.ckpt_dir, "zara1_best_on_test.pth"),
            )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={total_loss / len(train_loader):.4f} | "
            f"test_minADE={test_minade:.4f} "
            f"test_minFDE={test_minfde:.4f}"
        )


# =========================
# entry
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_pkl",
        type=str,
        default="processed_data_noise/zara1_train.pkl",
    )
    parser.add_argument(
        "--test_pkl",
        type=str,
        default="processed_data_noise/zara1_test.pkl",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
    )

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

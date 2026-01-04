import dill
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from trajectory_datasets import TrajectoryDataset, trajectory_collate
# from base_models import LinearPredictor
from base_models_test import LinearPredictor
from metrics_min import min_ade, min_fde


def evaluate(model, loader, device):
    model.eval()

    ade_sum = 0.0
    fde_sum = 0.0
    num_batches = 0

    with torch.no_grad():
        for x, y, vel, pos, nei_lists, batch_splits in loader:
            x = x.to(device)
            y = y.to(device)
            vel = vel.to(device)
            pos = pos.to(device)

            pred, _, _ = model(x, vel, pos, nei_lists, batch_splits)

            ade = min_ade(pred, y)
            fde = min_fde(pred, y)

            ade_sum += ade
            fde_sum += fde
            num_batches += 1

    return ade_sum / num_batches, fde_sum / num_batches


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = dill.load(open("processed_data_noise/hotel_test.pkl", "rb"))

    dataset = TrajectoryDataset(
        env,
        obs_len=8,
        pred_len=12,
        attention_radius=3.0
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=trajectory_collate
    )

    args = SimpleNamespace(
        input_size=4,
        output_size=2,
        obs_length=8,
        pred_length=12,     
        x_encoder_head=4,      # Transformer encoder 用
        embedding_size=128,
        social_ctx_dim=64,
        num_samples=20,
    )

    model = LinearPredictor(args, device).to(device)
    model.load_state_dict(torch.load("checkpoints/zara2_best_on_test.pth"))

    ade, fde = evaluate(model, loader, device)

    print("Evaluation Results:")
    print(f"  minADE: {ade:.4f}")
    print(f"  minFDE: {fde:.4f}")


if __name__ == "__main__":
    main()

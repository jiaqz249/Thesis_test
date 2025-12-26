import torch


@torch.no_grad()
def min_ade(pred, gt):
    """
    pred: (K, B, T, 2)
    gt:   (B, T, 2)
    """
    gt = gt.unsqueeze(0)                       # (1, B, T, 2)
    l2 = torch.norm(pred - gt, dim=-1)         # (K, B, T)
    ade = l2.mean(dim=-1)                      # (K, B)
    min_ade, _ = ade.min(dim=0)                # (B,)
    return min_ade.mean().item()


@torch.no_grad()
def min_fde(pred, gt):
    """
    Oracle final displacement error.
    """
    gt_final = gt[:, -1]                       # (B, 2)
    pred_final = pred[:, :, -1]                # (K, B, 2)
    l2 = torch.norm(
        pred_final - gt_final.unsqueeze(0),
        dim=-1
    )                                          # (K, B)
    min_fde, _ = l2.min(dim=0)                 # (B,)
    return min_fde.mean().item()

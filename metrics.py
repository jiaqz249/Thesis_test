import torch


@torch.no_grad()
def mean_ade(pred, gt):
    """
    pred: (K, B, T, 2)
    gt:   (B, T, 2)
    """
    gt = gt.unsqueeze(0)                       # (1, B, T, 2)
    l2 = torch.norm(pred - gt, dim=-1)         # (K, B, T)
    ade = l2.mean(dim=-1)                      # (K, B)
    return ade.mean().item()                   # over K and B


@torch.no_grad()
def mean_fde(pred, gt):
    """
    Mean final displacement error over K modes.
    """
    gt_final = gt[:, -1]                       # (B, 2)
    pred_final = pred[:, :, -1]                # (K, B, 2)
    l2 = torch.norm(pred_final - gt_final.unsqueeze(0), dim=-1)
    return l2.mean().item()                    # over K and B

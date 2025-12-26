import torch
import torch.nn.functional as F


def best_of_k_ade(pred, gt):
    gt_exp = gt.unsqueeze(0)
    l2 = torch.norm(pred - gt_exp, dim=-1)
    ade = l2.mean(dim=-1)
    min_ade, best_k = ade.min(dim=0)
    return min_ade.mean(), best_k


def mode_classification_loss(mode_logits, best_k):
    return F.cross_entropy(mode_logits, best_k)


def trajectory_diversity_loss(pred, max_dist=10.0):
    K = pred.size(0)
    diff = pred.unsqueeze(1) - pred.unsqueeze(0)
    dist = torch.norm(diff, dim=-1).mean(dim=(2, 3))
    dist = torch.clamp(dist, max=max_dist)
    mask = 1.0 - torch.eye(K, device=pred.device)
    return -(dist * mask).sum() / (K * (K - 1))


def mode_entropy_loss(mode_logits, eps=1e-6):
    prob = F.softmax(mode_logits, dim=-1)
    entropy = - (prob * torch.log(prob + eps)).sum(dim=1).mean()
    return entropy


def trajectory_loss(
    pred,
    gt,
    mode_logits,
    lambda_div,
    lambda_cls,
    lambda_ent,
):
    loss_reg, best_k = best_of_k_ade(pred, gt)
    loss_div = trajectory_diversity_loss(pred)
    loss_cls = mode_classification_loss(mode_logits, best_k)
    loss_ent = mode_entropy_loss(mode_logits)

    total_loss = (
        loss_reg
        + lambda_div * loss_div
        + lambda_cls * loss_cls
        + lambda_ent * loss_ent
    )
    return total_loss

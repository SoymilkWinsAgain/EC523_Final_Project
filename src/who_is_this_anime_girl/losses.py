from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(embeddings.device)

    logits = embeddings @ embeddings.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    logits_mask = torch.ones_like(positive_mask) - torch.eye(positive_mask.size(0), device=embeddings.device)
    positive_mask = positive_mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positives_per_anchor = positive_mask.sum(dim=1)
    valid = positives_per_anchor > 0
    if not torch.any(valid):
        return embeddings.sum() * 0.0

    mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positives_per_anchor.clamp_min(1.0)
    return -mean_log_prob[valid].mean()

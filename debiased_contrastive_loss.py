import torch
import torch.nn as nn
import torch.nn.functional as F

class DebiasedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, p=0.1):
        super().__init__()
        self.temperature = temperature
        self.p = p

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        device = z_i.device

        # Normalize
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # Stack representations
        reps = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        sim = torch.matmul(reps, reps.T)     # (2B, 2B)

        # Apply temperature
        logits = sim / self.temperature

        # Positive indices
        pos_indices = torch.arange(batch_size, 2*batch_size, device=device)
        pos_indices = torch.cat([pos_indices, torch.arange(batch_size, device=device)])

        # Positive similarities per sample
        positives = logits[torch.arange(2*batch_size, device=device), pos_indices]

        # Mask out self-pairs and positives for negatives
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.fill_diagonal_(False)
        mask[torch.arange(2*batch_size, device=device), pos_indices] = False

        # Debiased denominator
        exp_logits = torch.exp(logits)
        neg_sum = exp_logits * mask
        neg_sum = neg_sum.sum(dim=1)

        # Correct debiasing: subtract p * exp(positive) / (1-p)
        denom = neg_sum - (self.p / (1 - self.p)) * torch.exp(positives)

        # Loss
        loss = -torch.log(torch.exp(positives) / (torch.exp(positives) + denom))
        return loss.mean()


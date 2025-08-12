import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        device = z_i.device

        # Normalize embeddings
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # Concatenate representations
        representations = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # Cosine similarity
        similarity_matrix = torch.matmul(representations, representations.T)

        # Mask self-similarity
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Temperature scaling
        similarity_matrix = similarity_matrix / self.temperature

        # Positive pairs: z_i[k] → z_j[k], z_j[k] → z_i[k]
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size),
            torch.arange(batch_size)
        ]).to(device=device, dtype=torch.long)

        # Cross-entropy loss
        loss = self.criterion(similarity_matrix, labels)
        return loss


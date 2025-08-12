import torch
import torch.nn as nn
import torch.nn.functional as F

class DebiasedContrastiveLoss(nn.Module):
    """
    Calculates the Debiased Contrastive loss for self-supervised learning.
    This loss encourages the positive feature samples to be close to each other
    while pushing negative feature samples far apart. It includes a debiasing
    term to account for sampling bias.
    """
    def __init__(self, temperature=0.07, p=0.1):
        """
        Args:
            temperature (float, optional): The temperature scaling factor.
                Defaults to 0.07.
            p (float, optional): The debiasing factor, representing the probability
                of a sampled negative actually being a positive. Defaults to 0.1.
        """
        super().__init__()
        self.temperature = temperature
        self.p = p

    def forward(self, z_i, z_j):
        """
        Forward pass for the Debiased Contrastive loss.

        Args:
            z_i (torch.Tensor): The first batch of embeddings (e.g., view 1 of the data).
                                Shape: (batch_size, embedding_dim)
            z_j (torch.Tensor): The second batch of embeddings (e.g., view 2 of the data).
                                Shape: (batch_size, embedding_dim)

        Returns:
            torch.Tensor: The calculated Debiased Contrastive loss.
        """
        batch_size = z_i.shape[0]
        device = z_i.device

        # 1. Normalize the embeddings to have unit length
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # 2. Concat the two batches of embeddings
        representations = torch.cat([z_i, z_j], dim=0)
        num_samples = representations.shape[0]

        # 3. Calculate the cosine similarity between all pairs of embeddings
        similarity_matrix = torch.matmul(representations, representations.T)

        # 4. Create the mask for positive pairs.
        # The positive pair for the i-th embedding in the first half (z_i)
        # is the i-th embedding in the second half (z_j), which is at index (i + batch_size).
        # And vice-versa.
        labels = torch.cat([
            torch.arange(batch_size, num_samples),
            torch.arange(batch_size)
        ]).to(device)
        pos_mask = F.one_hot(labels, num_classes=num_samples).bool()

        # 5. Apply temperature scaling
        logits = similarity_matrix / self.temperature

        # 6. Get the logits for positive pairs
        positives = logits[pos_mask]

        # 7. Create mask for negative pairs.
        # Negatives are all pairs that are not positive and not self-pairs.
        neg_mask = ~pos_mask
        neg_mask.fill_diagonal_(False)

        # 8. Calculate the denominator with debiasing
        # The debiasing term down-weights the contribution of negatives.
        denom = torch.exp(logits) * neg_mask.float()
        denom = (1 - self.p) * denom.sum(dim=1)

        # 9. Calculate the final loss for each sample
        # The loss is the negative log-likelihood of the positive sample among a debiased set of negatives.
        loss = -torch.log(torch.exp(positives) / (torch.exp(positives) + denom))

        # 10. Return the mean loss over the batch
        return loss.mean()

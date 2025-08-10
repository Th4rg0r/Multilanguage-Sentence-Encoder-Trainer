import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This loss encourages the positive feature samples to be close to each other
    while pushing negative feature samples far apart.
    """
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature (float, optional):  The temperature scaling factor.
                a  smaller temparature makes the model more sensitive to hard negatives.
                Defaults to 0.07, a common value in literature.
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, z_i, z_j):
        """
        Forward pass for the InfoNCE loss.
        
        Args:
            z_i (torch.Tensor): The first  batch of embeddings (e.g., view 1 of the data).
                                Shape: (batch_size, embedding_dim)
            z_j (torch.Tensor): The second batch of embeddings (e.g., view 2 of the data).
                                Shape: (batch_size, embedding_dim)

        Returns:
            torch.Tensor: The calculated InfoNCE loss
        """

        batch_size =  z_i.shape[0]
        device =  z_i.device

        # 1.  Normalize the embeddings to have unit length
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        # 2. Concat the two batches of embeddings to create a larger batch 
        # of size  (2 * batch_size, embedding_dim)
        representations = torch.cat([z_i, z_j], dim=0)

        # 3. Calculate the cosine similarity between all pairs of embeddings
        # The resulting matrix will be of shape (2 * batch_size, 2 * batch_size)
        print (representations.shape)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # 4. Create a mask to remove  the similarity  of an  embedding with itself (the diagonal)
        # This is to prevent the model from trivial collapsing to a solution 
        # where it learns to  perfectly match an embedding  with itself.
        self_mask = torch.eye(2* batch_size, device=device, dtype=torch.bool)
        similarity_matrix.masked_fill_(self_mask, -9e15) # fill with a large negative number

        #5. Scale similarity matrix by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # 6. Create the labels for the cross-entropy loss.
        # The positive pair for the i-th embedding in the first half (z_i)
        # is the i-th embedding in the second half (z_j), which is at index (i + batch_size).
        # And vice-versa.
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size),
            torch.arange(batch_size)
        ]).to(device)

        # 7. Calculate the cross-entropy loss.
        # This one loss  calculation handles both directions (z_i -> z_j and z_j -> z_i)

        loss = self.criterion(similarity_matrix, labels)
    
        return loss
    

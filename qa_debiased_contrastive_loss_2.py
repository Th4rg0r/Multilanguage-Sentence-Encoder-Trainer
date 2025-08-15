import torch
import torch.nn as nn
import torch.nn.functional as F

class QADebiasedContrastiveLoss(nn.Module):
    """
    Contrastive loss for QA fine-tuning with debiased InfoNCE.
    
    Components:
    - QS Debiased InfoNCE: align each question with all sentences in its paragraph,
      using a debiased denominator (Chuang et al., 2020).
    - SQ InfoNCE: align each section title with questions from the same section (positives)
      versus questions from other sections (negatives).
    - AT InfoNCE: align article title with section titles in the same article. Skipped if
      no negatives (single-article setting).
    
    Args:
        tau (float): Temperature for contrastive softmax.
    """
    def __init__(self, tau: float = 0.05):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        article_title_embedding: torch.Tensor,          # shape: [D]
        section_title_embeddings: list[torch.Tensor],   # list of [D] (len = n_sections)
        question_embeddings_by_paragraph: list[list[torch.Tensor]],  # [[D],...] per paragraph
        sentence_embeddings_by_paragraph: list[list[torch.Tensor]],# [[D],...] per paragraph
        paragraph_idx_for_question: torch.Tensor, # [Q] flat tensor of global paragraph indices
        section_idx_for_question: torch.Tensor # [Q] flat tensor of section indices
    ) -> torch.Tensor:
        device = article_title_embedding.device
        tau = self.tau

        # Stack section-title embeddings: shape [S, D]
        if len(section_title_embeddings) > 0:
            section_embs = torch.stack(section_title_embeddings).to(device)  # [S, D]
        else:
            section_embs = torch.empty((0, article_title_embedding.size(0)), device=device)

        # Flatten all questions into one tensor, track section and paragraph indices
        question_embs_list = []
        # No need for paragraph_idx_for_question_flat here, use the input tensor directly
        
        for para_idx, question_list in enumerate(question_embeddings_by_paragraph): # Iterate through paragraphs
            for q_emb in question_list: # Iterate through questions in this paragraph
                question_embs_list.append(q_emb.squeeze(0)) # Squeeze the batch dimension
        
        if question_embs_list:
            question_embs = torch.stack(question_embs_list).to(device)  # [Q, D]
            # paragraph_idx_for_question is already a tensor, no need to recreate
        else:
            # No questions
            question_embs = torch.empty((0, section_embs.size(1) if section_embs.numel() else article_title_embedding.size(0)), device=device)
            # paragraph_idx_for_question is already handled by the input, no need to recreate empty

        # Flatten all sentences: shape [N_sents, D], track paragraph index per sentence
        sentence_embs_list = []
        sent_paragraph_idx = []
        for p_idx, sentence_list in enumerate(sentence_embeddings_by_paragraph):
            for s_emb in sentence_list:
                sentence_embs_list.append(s_emb.squeeze(0) if s_emb.dim() > 1 else s_emb)
                sent_paragraph_idx.append(p_idx)
        if sentence_embs_list:
            sentence_embs = torch.stack(sentence_embs_list).to(device)  # [N_sents, D]
            sent_paragraph_idx = torch.tensor(sent_paragraph_idx, dtype=torch.long, device=device)
        else:
            sentence_embs = torch.empty((0, question_embs.size(1) if question_embs.numel() else section_embs.size(1) if section_embs.numel() else article_title_embedding.size(0)), device=device)
            sent_paragraph_idx = torch.empty((0,), dtype=torch.long, device=device)

        # Normalize all embeddings to unit length
        if article_title_embedding.dim() == 1:
            article_emb = F.normalize(article_title_embedding.to(device), dim=0)  # [D]
        else:
            article_emb = F.normalize(article_title_embedding.to(device), dim=-1)
        if section_embs.numel() > 0:
            section_embs = F.normalize(section_embs, dim=1)  # [S, D]
        if question_embs.numel() > 0:
            question_embs = F.normalize(question_embs, dim=1)  # [Q, D]
        if sentence_embs.numel() > 0:
            sentence_embs = F.normalize(sentence_embs, dim=1)  # [N_sents, D]

        # Compute Question–Sentence Debiased InfoNCE Loss
        QS_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if question_embs.numel() > 0 and sentence_embs.numel() > 0:
            # Similarity matrix [Q x N_sents]
            sim_QS = torch.matmul(question_embs, sentence_embs.t()) / tau  # [Q, N_sents]
            num_paragraphs = len(sentence_embeddings_by_paragraph) or 1
            eta_plus = 1.0 / max(num_paragraphs, 1)
            eta_minus = 1.0 - eta_plus
            # Precompute exp(-1/tau) lower bound
            min_val = torch.exp(torch.tensor(-1.0 / tau, device=device))
            total_pairs = 0
            for q_idx in range(sim_QS.size(0)):
                # Directly use the tensor for paragraph index
                # paragraph_idx_for_question is the input tensor [Q]
                para_idx_for_current_q = paragraph_idx_for_question[q_idx] 
                
                # Indices of all sentences in this paragraph (positives)
                pos_mask = (sent_paragraph_idx == para_idx_for_current_q) # Compare tensor with tensor
                if not pos_mask.any():
                    continue
                pos_idxs = pos_mask.nonzero(as_tuple=True)[0]
                neg_mask = ~pos_mask
                neg_idxs = neg_mask.nonzero(as_tuple=True)[0]

                # Sum of exponentiated similarities for positives and negatives
                exp_sims = torch.exp(sim_QS[q_idx])
                sum_pos = exp_sims[pos_idxs].sum()
                sum_neg = exp_sims[neg_idxs].sum()
                N_neg = len(neg_idxs)
                M_pos = len(pos_idxs)
                if N_neg == 0 or M_pos == 0:
                    continue

                # Debiased term g = max{ (1/eta_minus)*(sum_neg/N_neg - (eta_plus * sum_pos)/M_pos ), exp(-1/tau) }
                avg_neg = sum_neg / N_neg
                avg_pos = sum_pos / M_pos
                val = (avg_neg - eta_plus * avg_pos) / eta_minus
                g_val = torch.max(val, min_val)
                # Accumulate loss for each positive sentence pair
                total_pairs += M_pos
                for pi in pos_idxs:
                    numer = torch.exp(sim_QS[q_idx, pi])
                    denom = numer + N_neg * g_val
                    QS_loss = QS_loss - torch.log(numer / denom)

            # Normalize by number of positive pairs
            if total_pairs > 0:
                QS_loss = QS_loss / total_pairs

        # Compute Section–Question InfoNCE Loss
        SQ_loss = torch.tensor(0.0, device=device, requires_grad=True)
        if section_embs.numel() > 0 and question_embs.numel() > 0:
            # Similarity matrix [S x Q]
            sim_SQ = torch.matmul(section_embs, question_embs.t()) / tau  # [S, Q]
            total_pairs = 0
            for s_idx in range(section_embs.size(0)):
                # Questions whose section index == s_idx
                # section_idx_for_question is the input tensor [Q]
                pos_mask_q = (section_idx_for_question == s_idx) # Compare tensor with scalar
                pos_q_idxs = pos_mask_q.nonzero(as_tuple=True)[0]
                if len(pos_q_idxs) == 0:
                    continue
                neg_mask_q = ~pos_mask_q
                neg_q_idxs = neg_mask_q.nonzero(as_tuple=True)[0]
                # Sum exponentials for negatives (questions from other sections)
                exp_sims = torch.exp(sim_SQ[s_idx])
                sum_neg = exp_sims[neg_q_idxs].sum() if len(neg_q_idxs) > 0 else torch.tensor(0.0, device=device)
                N_neg = len(neg_q_idxs)
                for qi in pos_q_idxs:
                    numer = torch.exp(sim_SQ[s_idx, qi])
                    # If no negatives, skip denominator (degenerate case)
                    if N_neg == 0:
                        continue
                    denom = numer + sum_neg
                    SQ_loss = SQ_loss - torch.log(numer / denom)
                    total_pairs += 1
            if total_pairs > 0:
                SQ_loss = SQ_loss / total_pairs

        # Compute Article–Section InfoNCE Loss (skip if no negatives, i.e., single article)
        AT_loss = torch.tensor(0.0, device=device)
        if section_embs.numel() > 0:
            # Align article title (as anchor) with section titles of same article.
            # If there were other articles, we would include them as negatives; here we assume single-article.
            # Thus, no valid negatives => loss = 0.
            pass  # AT_loss remains 0.0

        # Total loss
        total_loss = QS_loss + SQ_loss + AT_loss
        return total_loss

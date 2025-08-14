import torch
import torch.nn as nn
import torch.nn.functional as F

class QADebiasedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, debiasing_lambda=0.1):
        super(QADebiasedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.debiasing_lambda = debiasing_lambda

    def forward(self, article_title_embedding, section_title_embeddings,
                question_embeddings_by_section, sentence_embeddings_by_paragraph):
        
        total_loss = 0.0
        qs_loss = torch.tensor(0.0, device=article_title_embedding.device)
        sq_loss = torch.tensor(0.0, device=article_title_embedding.device)
        at_loss = torch.tensor(0.0, device=article_title_embedding.device)

        # 1. Question-Sentence Alignment Loss (Debiased Contrastive Loss)
        # For each question, its relevant sentences (those in the same paragraph) are treated as positive samples.
        # All other sentences (from different paragraphs or other articles) are negatives.
        
        # Flatten all sentence embeddings for negative sampling across the article
        all_sentences_in_article_embeddings = []
        for para_embeddings_list in sentence_embeddings_by_paragraph:
            if para_embeddings_list: # Check if the list is not empty
                all_sentences_in_article_embeddings.extend(para_embeddings_list)
        
        if all_sentences_in_article_embeddings:
            all_sentences_in_article_tensor = torch.cat(all_sentences_in_article_embeddings, dim=0) # (num_total_sentences, embedding_dim)
            
            for section_idx, section_questions_embeddings in enumerate(question_embeddings_by_section):
                for q_emb in section_questions_embeddings:
                    # Identify positive sentences for this question (sentences in the same paragraph)
                    # This assumes a direct mapping from section_idx to paragraph_idx for simplicity.
                    # In a real scenario, you'd need to know which paragraph a question belongs to.
                    # For now, let's assume sentences_by_paragraph[section_idx] contains sentences for the current question's context.
                    
                    # This part needs careful handling of positive and negative indices.
                    # For now, I'll assume that for a given question, all sentences in the same 'paragraph' (as represented by sentence_embeddings_by_paragraph[section_idx]) are positives.
                    # And all other sentences in the article are negatives.
                    
                    positive_sentence_embeddings = sentence_embeddings_by_paragraph[section_idx]
                    
                    if positive_sentence_embeddings:
                        positive_sentence_embeddings_tensor = torch.cat(positive_sentence_embeddings, dim=0)
                        
                        # Compute similarities between question and all sentences in the article
                        # Normalize embeddings
                        q_emb_norm = F.normalize(q_emb, p=2, dim=-1)
                        all_sentences_in_article_tensor_norm = F.normalize(all_sentences_in_article_tensor, p=2, dim=-1)

                        # Compute similarities using matrix multiplication
                        similarities_to_all_sentences = torch.matmul(q_emb_norm, all_sentences_in_article_tensor_norm.transpose(0, 1))
                        
                        # Identify positive similarities
                        # Normalize embeddings
                        q_emb_norm = F.normalize(q_emb, p=2, dim=-1)
                        positive_sentence_embeddings_tensor_norm = F.normalize(positive_sentence_embeddings_tensor, p=2, dim=-1)

                        # Compute similarities using matrix multiplication
                        pos_similarities = torch.matmul(q_emb_norm, positive_sentence_embeddings_tensor_norm.transpose(0, 1))
                        
                        individual_qs_losses = []
                        if pos_similarities.numel() > 0: # Check if there are any positive similarities
                            for pos_sim in pos_similarities:
                                pos = torch.exp(pos_sim / self.temperature)
                                
                                # Collect negative sentences from other sections within the same article
                                negative_sentences_in_article_embeddings = []
                                for other_section_idx, other_section_sentences in enumerate(sentence_embeddings_by_paragraph):
                                    if other_section_idx != section_idx and other_section_sentences:
                                        negative_sentences_in_article_embeddings.extend(other_section_sentences)
                                
                                if negative_sentences_in_article_embeddings:
                                    negative_sentences_in_article_tensor = torch.cat(negative_sentences_in_article_embeddings, dim=0)
                                    neg_similarities = torch.matmul(F.normalize(q_emb, p=2, dim=-1), F.normalize(negative_sentences_in_article_tensor, p=2, dim=-1).transpose(0, 1))
                                    neg_sum_exp = torch.sum(torch.exp(neg_similarities / self.temperature))
                                else:
                                    neg_sum_exp = torch.tensor(0.0, device=q_emb.device)
                                
                                # N_g calculation
                                N_e = all_sentences_in_article_tensor.size(0) # Total number of sentences in the article
                                
                                # The formula for N_g from the instructions: Ng=max(-λpos+neg/(1-λ), Ne-1/τ)
                                # Here, 'neg' is the sum of exp(neg_sims/tau)
                                
                                term1 = -self.debiasing_lambda * pos + neg_sum_exp / (1 - self.debiasing_lambda)
                                term2 = torch.exp(torch.tensor(-1.0 / self.temperature, device=q_emb.device)) * (N_e - 1) # N-1 for the number of negatives
                                
                                N_g = torch.max(term1, term2)
                                
                                # Ensure N_g is not zero or negative to avoid log(0) or division by zero
                                N_g = torch.clamp(N_g, min=1e-8) # Small positive value
                                
                                individual_qs_losses.append(-torch.log(pos / (pos + N_g)))
                            
                            if individual_qs_losses:
                                qs_loss += torch.sum(torch.stack(individual_qs_losses))
        
        # 2. Section Title-Question Alignment Loss (InfoNCE)
        sq_loss = 0.0
        all_questions_in_article_embeddings = []
        for section_q_emb_list in question_embeddings_by_section:
            if section_q_emb_list:
                all_questions_in_article_embeddings.extend(section_q_emb_list)
        
        if section_title_embeddings and all_questions_in_article_embeddings:
            all_questions_in_article_tensor = torch.cat(all_questions_in_article_embeddings, dim=0)
            
            for section_idx, s_title_emb in enumerate(section_title_embeddings):
                positive_questions_embeddings = question_embeddings_by_section[section_idx]
                
                if positive_questions_embeddings:
                    positive_questions_embeddings_tensor = torch.cat(positive_questions_embeddings, dim=0)
                    
                    # Compute similarities for positives
                    # Normalize embeddings
                    s_title_emb_norm = F.normalize(s_title_emb, p=2, dim=-1)
                    positive_questions_embeddings_tensor_norm = F.normalize(positive_questions_embeddings_tensor, p=2, dim=-1)

                    # Compute similarities using matrix multiplication
                    pos_similarities = torch.matmul(s_title_emb_norm, positive_questions_embeddings_tensor_norm.transpose(0, 1))
                    
                    # Negatives: questions from other sections within the same article
                    negative_questions_embeddings = []
                    for other_section_idx, other_section_q_emb in enumerate(question_embeddings_by_section):
                        if other_section_idx != section_idx and other_section_q_emb:
                            negative_questions_embeddings.extend(other_section_q_emb)
                    
                    if negative_questions_embeddings:
                        negative_questions_embeddings_tensor = torch.cat(negative_questions_embeddings, dim=0)
                        neg_similarities = torch.matmul(F.normalize(s_title_emb, p=2, dim=-1), F.normalize(negative_questions_embeddings_tensor, p=2, dim=-1).transpose(0, 1))
                    else:
                        neg_similarities = torch.empty(0, device=s_title_emb.device)
                    
                    # InfoNCE Loss
                    numerator = torch.sum(torch.exp(pos_similarities / self.temperature))
                    denominator = numerator + torch.sum(torch.exp(neg_similarities / self.temperature))
                    
                    if denominator > 0:
                        sq_loss += -torch.log(numerator / denominator)
        
        # 3. Article Title-Section Alignment Loss (InfoNCE)
        # This requires negatives from *other* articles. For a batch_size of 1, this is tricky.
        # For now, I'll implement it assuming a single article and no cross-article negatives.
        # A more complete solution would involve a memory bank or larger batches.
        
        at_loss = 0.0
        if article_title_embedding is not None and section_title_embeddings:
            section_title_embeddings_tensor = torch.cat(section_title_embeddings, dim=0)
            
            # Positives: all section titles within this article
            # Normalize embeddings
            article_title_embedding_norm = F.normalize(article_title_embedding, p=2, dim=-1)
            section_title_embeddings_tensor_norm = F.normalize(section_title_embeddings_tensor, p=2, dim=-1)

            # Positives: all section titles within this article
            pos_similarities = torch.matmul(article_title_embedding_norm, section_title_embeddings_tensor_norm.transpose(0, 1))
            
            # Negatives: (placeholder for now, as we only have one article in batch)
            # In a real scenario, these would come from other articles in the batch or a memory bank.
            neg_similarities = torch.empty(0, device=article_title_embedding.device) 
            
            numerator = torch.sum(torch.exp(pos_similarities / self.temperature))
            denominator = numerator + torch.sum(torch.exp(neg_similarities / self.temperature))
            
            if denominator > 0:
                at_loss += -torch.log(numerator / denominator)
        
        total_loss = qs_loss + sq_loss + at_loss
        return total_loss

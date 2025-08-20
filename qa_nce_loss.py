import torch
import torch.nn as nn
import torch.nn.functional as F

"""
A title
B title //other article

AA section
AB section // other sections in same article
BB section // other sections in different article

AAA question
AAA-2 question // other question for same parapgraph
AABquestion // other question for same section
ABBquestion // other questions for same article
BBBquestion // other questions for different article


AAAAanswer
AAABanswer // other answer for same questions
AABBanswer // other answers for same section
ABBBanswer // other answers for same article
BBBBanswer // other answers for different article

A title <=> AA section positive
A title <=> AAA question positive
A title <=> B title negative (very low p)
A title <=> BB section negative (very low p)
AA section <=> AAA question positive
AA section <=> AAAA answer  positive
AA section <=> AB section negative (high p)
AA section <=> BB section negative (very low p)
AA section <=> B title negative (low p)
AAA question <=> AAA-2 question posititive
AAA question <=> AAAA answer  positive
AAA question <=> ABB question  negative (high p)
AAA question <=> ABBB answer negative (low p)
AAAA answer <=> AAAB answer negative (high p)
AAAA answer <=> ABB question negative (high p)
"""

class ArticleLoss(nn.Module):
    def __init__(self, temperature=0.07, debiased_factor=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        self.p = debiased_factor
        
    def flatten_list(self, main_list):
        result_list = []
        for list_element in main_list:
            result_list.extend(list_element)
        return result_list
    
    def get_batches_for_article(self, index, article_list, section_list, question_list, answer_list):
        article_query = article_list[index]
        positives = []
        negatives = []
  
        positives.extend(section_list[index]) # sections in article positive
        positives.extend(self.flatten_list(self.flatten_list(question_list[index][:]))) # question in article positive
        negatives.extend(article_list[:index]+article_list[index+1:]) # other article titles negative
        negatives.extend(self.flatten_list(section_list[:index]+section_list[index+1:])) # other article section titles negative
        return article_query, positives, negatives
        
    def get_batches_for_section(self, article_index, section_index, article_list, section_list, question_list, answer_list):
        section_query = section_list[article_index][section_index]
        positives = []
        negatives = []
       
        positives.extend(self.flatten_list(question_list[article_index][section_index])) # questions in section positive
        positives.extend(self.flatten_list(answer_list[article_index][section_index])) # answers in section positive
        negatives.extend(section_list[article_index][:section_index] + section_list[article_index][section_index+1:]) # other section titles in article negative
        negatives.extend(self.flatten_list(section_list[:article_index] +section_list[article_index +1:]))  # other sections in other articles negative
        negatives.extend(articles[:article_index]+ articles[article_index+1:]) # other article titles negative
        return section_query, positives, negatives

    def get_batches_for_question(self, article_index, section_index,paragraph_index, question_index, article_list, section_list, question_list, answer_list):
        question_query = question_list[article_index][section_index][paragraph_index][question_index]
        positives = []
        negatives = []
        positives.extend(question_list[article_index][section_index][paragraph_index][:question_index]+question_list[article_index][section_index][paragraph_index][question_index+1:]) # same paragraph question positive
        positives.extend(answer_list[article_index][section_index][paragraph_index]) # same paragraph answers
        negatives.extend(self.flatten_list(question_list[article_index][section_index][:paragraph_index] + question_list[article_index][section_index][paragraph_index+1:])) #  same article other paragraph questions negative
        negatives.extend(self.flatten_list(self.flatten_list(answer_list[article_index][:section_index] + answer_list[article_index][section_index+1:])))
 # same article other section answers       
        negatives.extend(answer_list[article_index][section_index][:paragraph_index] + answer_list[article_index][section_index][paragraph_index+1])
        return question_query, positives, negatives

        
    def get_batches_for_answer(self,article_index, section_index, paragraph_index, answer_index, article_list, section_list, question_list, answer_list):
        answer_query = answer_list[article_index][section_index][paragraph_index][answer_index]
        positives = []
        negatives = []
        positives.extend(question_list[article_index][section_index][paragraph_index]) # same paragraph quesions positives 
        negatives.extend(self.flatten_list(self.flatten_list(self.flatten_list(answer_list[article_index][section_index][paragraph_index][:answer_index]+ answer_list[answer_index+1:])))) # same paragraph questions positive
        negatives.extend(self.flatten_list(self.flatten_list(question_list[article_index][:section_index]+ question_list[article_index][section_index+1:]))) # questions other sections negative
        negatives.extend(self.flatten_list(question_list[article_index][section_index][:paragraph_index] + question_list[article_index][section_index][paragraph_index+1:]))
        return answer_query, positives, negatives

    def forward(self, article_list, section_list, question_list, answer_list):
        for article_idx, article in article_list:
            negative_article_list = article_list[:article_idx] + article_list[article_idx+1:]
            negative_section_tree = section_list[:article_idx] + section_list[article_idx+1:]
            positive_section_list = section_list[article_idx]
            
            

    
    def debiased_info_nce_multi_pos (self, q, k_pos, k_neg):
        
        # q : [B, D]
        # k_pos: [B, P, D]
        # k_neg: [B, N, Dj
        # k_comp: [B, P+N, D]
        k_comb = torch.cat((k_pos, k_neg), dim=-2)
        
        # normalize vectors for cosine similarity
        k_comb = F.normalize(k_comb, dim=-1)
        q = F.normalize(q, dim = -1)

       
        # similarity_matrix: [B, P+N]
        # calculate cosine  similarity
        similarity_matrix = torch.einsum("bd,bmd->bm", q, k_comb) / self.temperature


        # numerator: [B, P]
        numerator = similarity_matrix[:, :k_pos.shape[-2]]

        # numerator: [B]
        numerator = torch.sum(torch.exp(numerator), dim=-1)


        # denominator: [B, P+N]
        denominator = torch.sum(torch.exp(similarity_matrix), dim=-1)

        # debiased contrastive loss
        result = -torch.log(numerator/(denominator-(self.p*numerator)))

        return result


if __name__ == "__main__":
    B = 4
    P = 5
    N = 6
    D = 512

    query = torch.randn(B, D)
    positives = torch.randn(B, P, D)
    negatives = torch.randn(B, N, D)

    criterion = ArticleLoss()
    
    articles = ["1A", "2A"]
    sections = [["1A 1S", "1A 2S"],["2A 1S", "2A 2S"]]
    #paragraphs = [[["1A 1S 1P", "1A 1S 2P"], ["1A 2S 1P", "1A 2S 2P"]],[["2A 1S 1P","2A 1S 2P"],["2A 2S 1P", "2A 2S 2P"]]]
    questions = [
                    [[["1A 1S 1P 1Q", "1A 1S 1P 2Q"], ["1A 1S 2P 1Q", "1A 1S 2P 2Q"]],[["1A 2S 1P 1Q", "1A 2S 1P 2Q"], ["1A 2S 2P 1Q", "1A 2S 2P 2Q"]]],
                    [[["2A 1S 1P 1Q", "2A 1S 1P 2Q"], ["2A 1S 2P 1Q", "2A 1S 2P 2Q"]],[["2A 2S 1P 1Q", "2A 2S 1P 2Q"], ["2A 2S 2P 1Q", "2A 2S 2P 2Q"]]],
                ]
    answers = [
                    [[["1A 1S 1P 1AW", "1A 1S 1P 2AW"], ["1A 1S 2P 1AW", "1A 1S 2P 2AW"]],[["1A 2S 1P 1AW", "1A 2S 1P 2AW"], ["1A 2S 2P 1AW", "1A 2S 2P 2AW"]]],
                    [[["2A 1S 1P 1AW", "2A 1S 1P 2AW"], ["2A 1S 2P 1AW", "2A 1S 2P 2AW"]],[["2A 2S 1P 1AW", "2A 2S 1P 2AW"], ["2A 2S 2P 1AW", "2A 2S 2P 2AW"]]],
                ]
    
    result_articles =  criterion.get_batches_for_article(0, articles, sections, questions, answers)
    result_sections = criterion.get_batches_for_section(0, 0, articles, sections, questions, answers)
    result_questions = criterion.get_batches_for_question(0, 0, 0, 0, articles, sections, questions, answers)
    result_answers = criterion.get_batches_for_answer(0, 0, 0, 0, articles, sections, questions, answers)
    print(f"result_articles:\nquery:{result_articles[0]}\npositives:{result_articles[1]}\nnegatives:{result_articles[2]}")
    print(f"result_sections:\nquery:{result_sections[0]}\npositives:{result_sections[1]}\nnegatives:{result_sections[2]}")
    print(f"result_questions:\nquery:{result_questions[0]}\npositives:{result_questions[1]}\nnegatives:{result_questions[2]}")
    print(f"result_answers:\nquery:{result_answers[0]}\npositives:{result_answers[1]}\nnegatives:{result_answers[2]}")
    #print(f"result:{criterion.debiased_info_nce_multi_pos(query, positives, negatives)}") # hiho
    




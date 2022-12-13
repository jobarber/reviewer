import torch
from sentence_transformers import SentenceTransformer, util

from peerreviewer import DEVICE


class CohesiveModel:

    def __init__(self, pretrained_path='all-mpnet-base-v2'):
        self.model = SentenceTransformer(pretrained_path).to(DEVICE)

    def predict(self, sentences):
        total_score = 0.
        index_cohesivenesses = []
        for index, sentence in enumerate(sentences):
            if index == 0:
                continue
            sentence_embedding = self.model.encode(sentence)
            context = [sentences[index - 1]]
            context_embedding = self.model.encode(context)
            similarities = util.pytorch_cos_sim(sentence_embedding, context_embedding)
            total_score += torch.sum(similarities).item()
            index_cohesivenesses.append({'cohesiveness': total_score / index,
                                         # index is correct here b/c n - 1 contexts
                                         'sentence_index': index})
        return index_cohesivenesses

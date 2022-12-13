import torch
from sentence_transformers import SentenceTransformer

from peerreviewer import DEVICE


class NoveltyDetector(torch.nn.Module):
    """
    Detects the novelty of a sentence or paragraph within a larger
    set of sentences or paragraphs.

    TODO: A little more work needs to be done in the final system,
      perhaps clustering or something similar.
    """

    def __init__(self, pretrained_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        super(NoveltyDetector, self).__init__()
        self.model = SentenceTransformer(pretrained_path).to(DEVICE)

    def forward(self, context_documents, document):

        context_embeddings = self.model.encode(context_documents)
        document_embeddings = self.model.encode([document])
        return torch.cosine_similarity(torch.tensor(document_embeddings),
                                       torch.tensor(context_embeddings),
                                       dim=-1)


if __name__ == '__main__':
    novelty_detector = NoveltyDetector()

    # Try an entirely novel document
    print(novelty_detector(['Machines often function as partners for meaning making.',
                            'Many people fail to see them as partners, however.',
                            'That means the people once again think too highly of themselves and'
                            ' not enough of the things and beings that make their lives possible.'],
                           document='Willows line the river bank for miles.'))

    # Try a related document
    print(novelty_detector(['Machines often function as partners for meaning making.',
                            'Many people fail to see them as partners, however.',
                            'That means the people once again think too highly of themselves and'
                            ' not enough of the things and beings that make their lives possible.'],
                           document='Many people view machines as partners.'))

import torch

from peerreviewer.models.novelty import NoveltyDetector


class TestEmotionDetector:
    """
    Test ``novelty.NoveltyDetector``.
    """

    texts = (['Machines often function as partners for meaning making.',
              'Many people fail to see them as partners, however.',
              'That means the people once again think too highly of themselves and'
              ' not enough of the things and beings that make their lives possible.'])
    novelty_detector = NoveltyDetector()

    def test_novelty_detector_forward(self):
        """
        Check to ensure the ``NoveltyDetector.forward`` produces the right shape.
        """
        outputs = self.novelty_detector(self.texts,
                                        document='This is a test.')
        assert outputs.shape == torch.Size([3])

    def test_novelty_detector_novel(self):
        """
        Test whether the novelty detector can detect an obvious dissimilarity.
        """
        similarities = self.novelty_detector(self.texts,
                                             document='Willows line the river bank for miles.')
        assert torch.all(similarities < 0.01)

    def test_novelty_detector_not_novel(self):
        """
        Test whether the novelty detector can detect an obvious similarity.
        """
        similarities = self.novelty_detector(self.texts,
                                             document='Many people view machines as partners.')
        assert similarities[0] > 0.75
        assert similarities[1] > 0.3
        assert torch.all(similarities > 0.10)

import torch

from peerreviewer.models.summarizer import Summarizer


class TestSummarizer:

    text = ("Machine learning is the study of computer algorithms that can "
            "improve automatically through experience and by the use of data. "
            "It is seen as a part of artificial intelligence. Machine learning "
            "algorithms build a model based on sample data, known as 'training "
            "data', in order to make predictions or decisions without being "
            "explicitly programmed to do so. Machine learning algorithms are "
            "used in a wide variety of applications, such as in medicine, email "
            "filtering, speech recognition, and computer vision, where it is "
            "difficult or unfeasible to develop conventional algorithms to "
            "perform the needed tasks.")
    summarizer = Summarizer()

    def test_summarizer_forward(self):
        outputs = self.summarizer(self.text)
        assert outputs.logits.shape == torch.Size([1, 110, 50264])

    def test_summarizer_generation(self):
        generation = self.summarizer.generate(self.text)
        assert generation.startswith('</s><s>')

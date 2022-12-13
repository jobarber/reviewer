import torch

from peerreviewer.models.emotion import EmotionDetector


class TestEmotionDetector:

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
    emotion_detector = EmotionDetector()

    def test_emotion_detector_forward(self):
        outputs = self.emotion_detector(self.text)
        assert outputs.logits.shape == torch.Size([1, 6])

    def test_emotion_detector_predict_label(self):
        label_prob = self.emotion_detector.predict_label(self.text)
        assert label_prob['label'] in self.emotion_detector.labels.values()
        assert label_prob['probability'] > 0.5

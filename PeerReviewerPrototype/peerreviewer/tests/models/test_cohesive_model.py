import re

from peerreviewer.models.cohesiveness import CohesiveModel


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

    cohesive_text = ("Machine learning is the study of computer algorithms that can "
                     "improve automatically through experience and by the use of data. "
                     "It is seen as a part of artificial intelligence. Machine learning "
                     "algorithms build a model based on sample data, known as 'training "
                     "data', in order to make predictions or decisions without being "
                     "explicitly programmed to do so. Machine learning algorithms are "
                     "used in a wide variety of applications, such as in medicine, email "
                     "filtering, speech recognition, and computer vision, where it is "
                     "difficult or unfeasible to develop conventional algorithms to "
                     "perform the needed tasks.")

    noncohesive_text = ("Machine learning is the study of computer algorithms that can "
                        "improve automatically through experience and by the use of data. "
                        "Dump trucks are important pieces of the puzzle. Machine learning "
                        "algorithms build a model based on sample data, known as 'training "
                        "data', in order to make predictions or decisions without being "
                        "explicitly programmed to do so. Cakes are delicious.")

    cohesive_model = CohesiveModel()

    def test_cohesive_prediction(self):
        # find all sentences
        sentences = re.findall(r'(?:\s*).*?\.', self.cohesive_text)
        # find cohesiveness between each successive pair of sentences
        cohesive_sentences = self.cohesive_model.predict(sentences)
        assert cohesive_sentences[0]['cohesiveness'] > 0.4

    def test_noncohesive_prediction(self):
        # find all sentences
        sentences = re.findall(r'(?:\s*).*?\.', self.noncohesive_text)
        # find cohesiveness between each successive pair of sentences
        cohesive_sentences = self.cohesive_model.predict(sentences)
        assert cohesive_sentences[0]['cohesiveness'] < 0.05

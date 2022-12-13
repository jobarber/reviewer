import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from peerreviewer import DEVICE


class EmotionDetector(torch.nn.Module):

    def __init__(self, pretrained_path='bhadresh-savani/distilbert-base-uncased-emotion'):
        super(EmotionDetector, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_path).to(DEVICE)
        self.labels = {0: "sadness",
                       1: "joy",
                       2: "love",
                       3: "anger",
                       4: "fear",
                       5: "surprise"}

    def forward(self, documents, labels=None):
        tokenized = self.tokenizer(documents, return_tensors='pt', padding=True).to(DEVICE)
        return self.model(labels=labels, **tokenized)

    def predict_label(self, document):
        output = self([document])
        softmax = torch.softmax(output.logits[0], dim=-1)
        argmax = torch.argmax(softmax, dim=-1).item()
        return {'label': self.labels[argmax], 'probability': softmax[argmax]}


if __name__ == '__main__':
    emotion_detector = EmotionDetector()
    print(emotion_detector(['You are unbelievable!']))

import pytest
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForTokenClassification

from peerreviewer.trainers.pytorch_trainer import ModelTrainer


@pytest.mark.skip(reason="Test this locally. It hangs the GitHub tests.")
class TestPyTorchTrainer:

    def test_sequence_classification_trainer(self):
        trainer = ModelTrainer(hf_model=AutoModelForSequenceClassification,
                               hf_tokenizer=AutoTokenizer,
                               pretrained_model_path='distilbert-base-uncased',
                               dataset=load_dataset('imdb'),
                               learning_rate=5e-5,
                               seed=42,
                               num_samples=30,
                               metrics=['accuracy', 'precision', 'recall', 'f1'],
                               total_num_checkpoints=0,
                               num_labels=2)
        eval_metrics = trainer.train('imdb')
        assert eval_metrics['eval_loss'] < 1.
        assert eval_metrics['eval_accuracy']['accuracy'] > 0.5
        with torch.no_grad():
            tokenized = trainer.tokenizer(['This movie is amazing!'], return_tensors='pt').to(
                'cuda' if torch.cuda.is_available() else 'cpu')
            output = torch.softmax(trainer.model(**tokenized).logits, dim=-1)
            assert torch.argmax(output[0]) == 1

    def test_token_classification_trainer(self):
        trainer = ModelTrainer(hf_model=AutoModelForTokenClassification,
                               hf_tokenizer=AutoTokenizer,
                               pretrained_model_path='distilbert-base-uncased',
                               dataset=load_dataset('wnut_17'),
                               learning_rate=5e-5,
                               seed=42,
                               num_samples=100,
                               metrics=['accuracy', 'precision', 'recall', 'f1'],
                               total_num_checkpoints=0,
                               sample_name='tokens',
                               label_name='ner_tags',
                               num_labels=13)
        # get full label list
        label_list = trainer.dataset["train"].features[f"ner_tags"].feature.names
        eval_metrics = trainer.train('conll2003')
        assert eval_metrics['eval_loss'] < 0.5
        assert eval_metrics['eval_accuracy']['accuracy'] > 0.9
        with torch.no_grad():
            tokenized = trainer.tokenizer(['Obama sent aid to the United Arab Emirates .'],
                                          return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
            output = torch.softmax(trainer.model(**tokenized).logits, dim=-1)
            assert output[0][0][label_list.index('B-person')] > 0.01

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, DataCollatorForTokenClassification)


class ModelTrainer:
    """
    Trains models of all varieties.
    """
    file_path = Path(__file__)

    def __init__(self,
                 hf_model=AutoModelForSequenceClassification,
                 hf_tokenizer=AutoTokenizer,
                 pretrained_model_path="bert-base-cased",
                 dataset=load_dataset("imdb"),
                 learning_rate=5e-5,
                 seed=42,
                 num_samples=None,
                 metrics=['accuracy', 'precision', 'recall', 'f1', 'matthews_correlation'],
                 total_num_checkpoints=5,
                 sample_name='text',
                 label_name='label',
                 **model_kwargs):
        self.hf_model = hf_model
        self.model = hf_model.from_pretrained(pretrained_model_path, **model_kwargs)
        self.tokenizer = hf_tokenizer.from_pretrained(pretrained_model_path)
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.seed = seed
        self.num_samples = num_samples
        self.metrics = {metric: load_metric(metric) for metric in metrics}
        self.save_total_limit = total_num_checkpoints
        self.sample_name = sample_name
        self.label_name = label_name
        self.token_model = True if 'token' in self.hf_model.__name__.lower() else False

    def train(self, model_identifier):
        cwd = self.file_path.cwd()
        parent = cwd.parent
        while parent.name != 'peerreviewer':
            parent = parent.parent

        training_args = TrainingArguments(f"{parent}/modelcheckpoints/{model_identifier}",
                                          evaluation_strategy="epoch",
                                          learning_rate=self.learning_rate,
                                          save_strategy='epoch',
                                          save_total_limit=self.save_total_limit,
                                          seed=self.seed)
        # check if a token architecture
        if self.token_model:
            tokenized_datasets = self.dataset.map(self._tokenize_and_align_labels, batched=True)
            data_collator = DataCollatorForTokenClassification(self.tokenizer)
        else:
            tokenized_datasets = self.dataset.map(self._tokenize_function, batched=True)
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        train_dataset = tokenized_datasets["train"].shuffle(seed=self.seed)
        eval_dataset = tokenized_datasets["test"].shuffle(seed=self.seed)
        if self.num_samples is not None:
            train_dataset = train_dataset.select(range(self.num_samples))
            eval_dataset = eval_dataset.select(range(self.num_samples))
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
            data_collator=data_collator
        )
        trainer.train()
        eval_metrics = trainer.evaluate()
        trainer.save_model(f"{parent}/modelcheckpoints/{model_identifier}_final")
        self.tokenizer.save_pretrained(f"{parent}/modelcheckpoints/{model_identifier}_final")
        return eval_metrics

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), axis=-1).flatten()
        labels = torch.tensor(labels).flatten()
        relevant_predictions = predictions[torch.nonzero(labels != -100, as_tuple=True)[0]]
        relevant_labels = labels[torch.nonzero(labels != -100, as_tuple=True)[0]]
        computed_metrics = {}
        for metric in self.metrics:
            if metric in ('accuracy', 'matthews_correlation'):
                computed_metrics[metric] = self.metrics[metric].compute(predictions=relevant_predictions,
                                                                        references=relevant_labels)
            else:
                computed_metrics[metric] = self.metrics[metric].compute(predictions=relevant_predictions,
                                                                        references=relevant_labels,
                                                                        average='weighted')
        return computed_metrics

    def _tokenize_function(self, examples):
        batch = [' '.join(example) if isinstance(example, (list, tuple)) else example for example in
                 examples[self.sample_name]]
        return self.tokenizer(batch, padding="max_length", truncation=True)

    def _tokenize_and_align_labels(self, examples):
        """
        This function comes mostly from this post:

        https://huggingface.co/docs/transformers/custom_datasets#tok_ner
        """
        tokenized_inputs = self.tokenizer(examples[self.sample_name], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[self.label_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


if __name__ == '__main__':
    trainer = ModelTrainer(hf_model=AutoModelForSequenceClassification,
                           hf_tokenizer=AutoTokenizer,
                           pretrained_model_path="bert-base-cased",
                           dataset=load_dataset("imdb"),
                           learning_rate=5e-5,
                           seed=42,
                           num_samples=1_000,
                           metrics=['accuracy', 'precision', 'recall', 'f1', 'matthews_correlation'],
                           total_num_checkpoints=0,
                           num_labels=2)
    trainer.train('imdb')

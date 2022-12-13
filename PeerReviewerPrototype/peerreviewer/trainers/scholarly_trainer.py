from transformers import AutoModelForSequenceClassification, AutoTokenizer

from peerreviewer.dataloaders.dataloader import get_dataloader
from peerreviewer.trainers.pytorch_trainer import ModelTrainer


if __name__ == '__main__':
    dataset = get_dataloader(file_format='csv',
                             train='academic_theology_train',
                             test='academic_theology_test')
    trainer = ModelTrainer(hf_model=AutoModelForSequenceClassification,
                           hf_tokenizer=AutoTokenizer,
                           pretrained_model_path="bert-base-cased",
                           dataset=dataset,
                           learning_rate=5e-5,
                           seed=42,
                           num_samples=None,
                           metrics=['accuracy', 'precision', 'recall', 'f1', 'matthews_correlation'],
                           total_num_checkpoints=5,
                           sample_name='sentence',
                           label_name='label')
    trainer.train('academic_theology')

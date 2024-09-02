import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, AutoTokenizer
from enhanced_model import EnhancedBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import os
import json

class PhishingDataset(Dataset):
    def __init__(self, data_path, use_additional_features=True):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.use_additional_features = use_additional_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        labels = torch.tensor(item['label'])
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        if self.use_additional_features:
            additional_features = torch.tensor(item['tokenized_subject'] + [item['encoded_sender'], item['encoded_receiver']])
            result['additional_features'] = additional_features
        return result

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

print("Loading and preprocessing data...")
# Load preprocessed data
train_data_path = "E:/PG-dissertation/dataset/train_data.json"
test_data_path = "E:/PG-dissertation/dataset/test_data.json"

# Initialise tokeniser
tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, low_cpu_mem_usage=False)

# Create DataLoaders
dataset_with_features_train = PhishingDataset(train_data_path, use_additional_features=True)
dataset_with_features_test = PhishingDataset(test_data_path, use_additional_features=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataloader_with_features_train = DataLoader(dataset_with_features_train, batch_size=16, shuffle=True, collate_fn=data_collator, pin_memory=True)
dataloader_with_features_test = DataLoader(dataset_with_features_test, batch_size=16, shuffle=False, collate_fn=data_collator, pin_memory=True)

print(f"Training set size: {len(dataset_with_features_train)}")
print(f"Test set size: {len(dataset_with_features_test)}")

print("Initialising models...")
model_with_features = EnhancedBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_with_features.config.id2label = {0: 'SAVE EMAIL', 1: 'PHISHING EMAIL'}

training_args_with_features = TrainingArguments(
    output_dir='E:/PG-dissertation/results/enhanced_model_with_features',
    num_train_epochs=3,  # increase training rounds
    per_device_train_batch_size=16,
    save_steps=5_000,
    save_total_limit=2,
    logging_dir='E:/PG-dissertation/logs_with_features',
    logging_steps=50,
    fp16=True,
    learning_rate=3e-5,  # set learning rate
    weight_decay=0.01
)

print("Setting up trainer...")
trainer_with_features = Trainer(
    model=model_with_features,
    args=training_args_with_features,
    train_dataset=dataset_with_features_train,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training for model with features...")
start_time = time.time()
trainer_with_features.train()
training_time_with_features = time.time() - start_time
print("Training time for model with features:", training_time_with_features, "seconds")

# Save the model and tokeniser
print("Saving model and tokeniser...")
model_with_features.save_pretrained('E:/PG-dissertation/results/enhanced_model_with_features')
tokenizer.save_pretrained('E:/PG-dissertation/results/enhanced_model_with_features')

# Perform evaluation
print("Starting evaluation for model with features...")
eval_results_with_features = trainer_with_features.evaluate(eval_dataset=dataset_with_features_test)

print("Training completed and model saved.")
output_dir = 'E:/PG-dissertation/dataset/test-result'
os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/test_with_features.txt', 'w') as f:
    f.write(f"Model: Enhanced model with features\n")
    f.write(f"Training time: {training_time_with_features} seconds\n")
    f.write("Evaluation results:\n")
    for key, value in eval_results_with_features.items():
        f.write(f"  {key}: {value}\n")

print("Results saved to E:/PG-dissertation/dataset/test-result/test_with_features.txt")
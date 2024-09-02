import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import os
import json

class PhishingDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        labels = torch.tensor(item['label'])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

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
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True, low_cpu_mem_usage=False)

# Create DataLoaders
dataset_body_only_train = PhishingDataset(train_data_path)
dataset_body_only_test = PhishingDataset(test_data_path)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataloader_body_only_train = DataLoader(dataset_body_only_train, batch_size=16, shuffle=True, collate_fn=data_collator, pin_memory=True)
dataloader_body_only_test = DataLoader(dataset_body_only_test, batch_size=16, shuffle=False, collate_fn=data_collator, pin_memory=True)

print(f"Training set size: {len(dataset_body_only_train)}")
print(f"Test set size: {len(dataset_body_only_test)}")

print("Initialiing models...")
model_body_only = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_body_only.config.id2label = {0: 'SAVE EMAIL', 1: 'PHISHING EMAIL'}

training_args_body_only = TrainingArguments(
    output_dir='E:/PG-dissertation/results/model_body_only',
    num_train_epochs=3,                                 # training rounds
    per_device_train_batch_size=16,                     # batch size
    save_steps=5_000,                                   # Saving frequency
    save_total_limit=2,                                 # limitaion of checkpoints retained
    logging_dir='E:/PG-dissertation/logs_body_only',
    logging_steps=50,                                   # Frequency of logging records
    fp16=True,                                          # Enable mixed precision training
    learning_rate=3e-5,                                 # leaning rate
    weight_decay=0.01                                   # weight decay
)

print("Setting up trainer...")
trainer_body_only = Trainer(
    model=model_body_only,
    args=training_args_body_only,
    train_dataset=dataset_body_only_train,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training for body-only model...")
start_time = time.time()
trainer_body_only.train()
training_time_body_only = time.time() - start_time
print("Training time for body-only model:", training_time_body_only, "seconds")

# Save the model and tokeniser
print("Saving model and tokeniser...")
model_body_only.save_pretrained('E:/PG-dissertation/results/model_body_only')
tokenizer.save_pretrained('E:/PG-dissertation/results/model_body_only')

# Perform evaluation
print("Starting evaluation for body-only model...")
eval_results_body_only = trainer_body_only.evaluate(eval_dataset=dataset_body_only_test)

print("Training completed and model saved.")
output_dir = 'E:/PG-dissertation/dataset/test-result'
os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/test_body_only.txt', 'w') as f:
    f.write(f"Model: Body-only model\n")
    f.write(f"Training time: {training_time_body_only} seconds\n")
    f.write("Evaluation results:\n")
    for key, value in eval_results_body_only.items():
        f.write(f"  {key}: {value}\n")

print("Results saved to E:/PG-dissertation/dataset/test-result/test_body_only.txt")
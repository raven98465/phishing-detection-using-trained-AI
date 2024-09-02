from torch import nn
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, AutoTokenizer, BertModel, BertPreTrainedModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import os
import json

# Custom model class
class EnhancedBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + 64, config.num_labels)  # Adjust input size accordingly

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, tokenized_subject=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        if tokenized_subject is not None:
            combined_output = torch.cat((pooled_output, tokenized_subject), dim=1)
        else:
            combined_output = pooled_output

        logits = self.classifier(combined_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

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
        tokenized_subject = torch.tensor(item['tokenized_subject'])
        labels = torch.tensor(item['label'])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tokenized_subject': tokenized_subject,
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
dataset_with_subject_train = PhishingDataset(train_data_path)
dataset_with_subject_test = PhishingDataset(test_data_path)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataloader_with_subject_train = DataLoader(dataset_with_subject_train, batch_size=16, shuffle=True, collate_fn=data_collator, pin_memory=True)
dataloader_with_subject_test = DataLoader(dataset_with_subject_test, batch_size=16, shuffle=False, collate_fn=data_collator, pin_memory=True)

print(f"Training set size: {len(dataset_with_subject_train)}")
print(f"Test set size: {len(dataset_with_subject_test)}")

print("Initialising models...")
model_with_subject = EnhancedBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_with_subject.config.id2label = {0: 'SAVE EMAIL', 1: 'PHISHING EMAIL'}

training_args_with_subject = TrainingArguments(
    output_dir='E:/PG-dissertation/results/model_with_subject',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=5_000,
    save_total_limit=2,
    logging_dir='E:/PG-dissertation/logs_with_subject',
    logging_steps=50,
    fp16=True,
    learning_rate=3e-5,
    weight_decay=0.01
)

print("Setting up trainer...")
trainer_with_subject = Trainer(
    model=model_with_subject,
    args=training_args_with_subject,
    train_dataset=dataset_with_subject_train,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training for model with subject...")
start_time = time.time()
trainer_with_subject.train()
training_time_with_subject = time.time() - start_time
print("Training time for model with subject:", training_time_with_subject, "seconds")

# Save the model and tokeniser
print("Saving model and tokeniser...")
model_with_subject.save_pretrained('E:/PG-dissertation/results/model_with_subject')
tokenizer.save_pretrained('E:/PG-dissertation/results/model_with_subject')

# Perform evaluation
print("Starting evaluation for model with subject...")
eval_results_with_subject = trainer_with_subject.evaluate(eval_dataset=dataset_with_subject_test)

print("Training completed and model saved.")
output_dir = 'E:/PG-dissertation/dataset/test-result'
os.makedirs(output_dir, exist_ok=True)
with open(f'{output_dir}/test_with_subject.txt', 'w') as f:
    f.write(f"Model: Model with subject\n")
    f.write(f"Training time: {training_time_with_subject} seconds\n")
    f.write("Evaluation results:\n")
    for key, value in eval_results_with_subject.items():
        f.write(f"  {key}: {value}\n")

print("Results saved to E:/PG-dissertation/dataset/test-result/test_with_subject.txt")
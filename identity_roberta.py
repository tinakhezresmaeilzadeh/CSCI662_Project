import torch
from transformers import RobertaTokenizer, RobertaModel, TrainingArguments, Trainer, RobertaConfig, RobertaPreTrainedModel
from torch.utils.data import Dataset
import json
import numpy as np
from torch import nn

# Step 1: Load and Preprocess Data
class HateSpeechDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        self.tokenizer = tokenizer
        self.data = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                if entry["fold"] == "train":
                    self.data.append((entry['text'], entry['hate']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, padding='max_length', return_tensors='pt', truncation=True)
        inputs = {key: val.flatten() for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(1 if label else 0, dtype=torch.long)
        return inputs
                

class HateSpeechTestDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        self.tokenizer = tokenizer
        self.data = []
        
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line)
                
                if entry["fold"] == "test":
                    self.data.append((entry['text'], entry['hate']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        inputs = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, padding='max_length', return_tensors='pt', truncation=True)
        inputs = {key: val.flatten() for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(1 if label else 0, dtype=torch.long)
        return inputs

# Step 2: Tokenize the Data
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer.pad_token = tokenizer.eos_token

# Initialize dataset
dataset = HateSpeechDataset(tokenizer, 'identity_hate_corpora.jsonl')
test_dataset = HateSpeechTestDataset(tokenizer, 'identity_hate_corpora.jsonl')
print(dataset.data[0])

print("Dataset has been loaded...")

# Step 3: Initialize RoBERTa Model
#model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
class RobertaForBinaryClassification(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = super().forward(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state

        # Apply mean pooling on the sequence of hidden states
        mean_last_hidden_state = torch.mean(last_hidden_states, dim=1)
        logits = self.classifier(mean_last_hidden_state)

        

        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits

        

# Example usage
config = RobertaConfig.from_pretrained('roberta-base')
model = RobertaForBinaryClassification(config)




print("Model is loaded...")

# Step 4: Training Setup
training_args = TrainingArguments(
    output_dir='./roberta_finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    logging_steps=200,
    save_steps=10000,
    evaluation_strategy='steps',
    eval_steps=10000,
    load_best_model_at_end=True,
    optim='adamw_torch'
)

def compute_accuracy(pred):
    logits, labels = pred.predictions, pred.label_ids
    preds = np.argmax(logits, axis=-1)
    correct = np.sum(preds == labels)
    accuracy = correct / len(labels)
    return {"accuracy": accuracy}

# Step 5: Train the Model

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_accuracy
)

print("Model is being trained...")
trainer.train()
print("Model has been trained...")


# Step 6: Save the Model
model.save_pretrained('./roberta_finetuned')



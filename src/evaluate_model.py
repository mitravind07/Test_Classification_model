import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from sklearn.metrics import classification_report
import numpy as np

# Step 1: Load Tokenizer and Model
model_path = "models/final_distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Step 2: Load Dataset
dataset = load_dataset("imdb")

# Step 3: Tokenize Test Set
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

encoded_dataset = dataset.map(tokenize_function, batched=True)

# Step 4: Prepare DataLoader
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
test_dataloader = DataLoader(encoded_dataset["test"], batch_size=8, collate_fn=data_collator)

# Step 5: Evaluation Loop
model.eval()
preds = []
labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label = batch['label']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1)

        preds.extend(pred.tolist())
        labels.extend(label.tolist())

# Step 6: Metrics
print("Classification Report:")
print(classification_report(labels, preds))

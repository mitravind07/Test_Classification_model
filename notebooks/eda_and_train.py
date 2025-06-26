# eda_and_train.ipynb (Jupyter Notebook)

# Step 1: Ensure Required Libraries Are Installed
# Note: In sandboxed or restricted environments, use terminal or requirements.txt to install manually.
# Install manually in your terminal: pip install torch transformers datasets scikit-learn

# Step 2: Load Dataset
# If `datasets` is unavailable, fall back to a local CSV-based dataset.
try:
    from datasets import load_dataset
    raw_datasets = load_dataset("imdb")
except ModuleNotFoundError:
    import pandas as pd
    from sklearn.model_selection import train_test_split

    print("'datasets' module not found. Falling back to CSV dataset.")
    df = pd.read_csv("imdb_small.csv")  # Make sure to provide this file
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    train_texts, test_texts, train_labels, test_labels = train_test_split(df["text"], df["label"], test_size=0.2)

    raw_datasets = {
        "train": [{"text": t, "label": l} for t, l in zip(train_texts, train_labels)],
        "test": [{"text": t, "label": l} for t, l in zip(test_texts, test_labels)],
    }

print("Dataset loaded.")

# Step 3: Tokenization
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Apply tokenization
if isinstance(raw_datasets, dict):
    from datasets import Dataset
    train_dataset = Dataset.from_list(raw_datasets["train"]).map(tokenize_function, batched=True)
    test_dataset = Dataset.from_list(raw_datasets["test"]).map(tokenize_function, batched=True)
else:
    encoded_datasets = raw_datasets.map(tokenize_function, batched=True)
    train_dataset = encoded_datasets["train"]
    test_dataset = encoded_datasets["test"]

# Step 4: Prepare DataLoaders
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

# Step 5: Load Pretrained Model
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 6: Define Trainer
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="models/distilbert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 7: Train Model
trainer.train()

# Step 8: Evaluate Model
trainer.evaluate()

# Optional: Print example predictions
from sklearn.metrics import classification_report
import numpy as np

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

print(classification_report(labels, preds))

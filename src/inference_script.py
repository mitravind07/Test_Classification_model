# inference.py
# This script allows inference on custom text inputs using your fine-tuned model.

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Path to your fine-tuned model
model_path = "models/distilbert"  # Make sure this folder exists and contains model/tokenizer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Put model in eval mode
model.eval()

# Sample texts for prediction
sample_texts = [
    "The movie was absolutely wonderful and touching!",
    "I hated every second of it. It was a disaster."
]

# Tokenize
inputs = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    preds = torch.argmax(probs, dim=1)

# Print results
for text, pred, prob in zip(sample_texts, preds, probs):
    sentiment = "Positive" if pred.item() == 1 else "Negative"
    confidence = prob[pred].item()
    print(f"\nText: {text}\nPrediction: {sentiment}\nConfidence: {confidence:.4f}")

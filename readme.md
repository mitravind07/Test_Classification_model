# README.md

# 📝 Text Classification with DistilBERT

This project demonstrates how to build a text classification pipeline using a pre-trained transformer model from Hugging Face on the IMDB sentiment dataset. The pipeline includes tokenization, model fine-tuning, evaluation, and inference.

---

## 📁 Folder Structure

```
project-root/
├── notebooks/
│   └── eda_and_train.ipynb        # EDA + training notebook
├── src/
│   ├── data_loader.py             # Dataset loading (optional for scripting)
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── inference.py               # Run predictions on custom input
├── models/
│   └── distilbert/                # Fine-tuned model artifacts
├── reports/
│   └── submission.md              # Project summary and insights
├── requirements.txt               # Required packages
└── README.md                      # This file
```

---

## ⚙️ Setup Instructions

### 1. Create Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

### 2. Install Required Libraries
```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, you can manually install:
```bash
pip install torch transformers datasets scikit-learn
```

### 3. Download Dataset (IMDB)
No need to manually download; it's automatically fetched using `datasets.load_dataset("imdb")`.

---

## 🚀 Run the Project

### ➤ Option 1: Run All Steps in Notebook
Open and run `notebooks/eda_and_train.ipynb` to execute tokenization, training, evaluation, and inference in one place.

### ➤ Option 2: Use Python Scripts

#### 1. Train the Model
```bash
python src/train.py
```

#### 2. Evaluate the Model
```bash
python src/evaluate.py
```

#### 3. Make Predictions
```bash
python src/inference.py
```

---

## 🧪 Evaluation
- Evaluation metrics include accuracy, precision, recall, and F1-score using `sklearn`.
- Results are printed after running `evaluate.py`.

---

## 🌐 Multilingual Extension (Bonus)
To make the project multilingual:
- Replace model: use `xlm-roberta-base`
- Use dataset: `amazon_reviews_multi`
- Update tokenizer and training logic accordingly

---

## 👩‍💻 Author
ML/NLP Engineer Intern Project

---

## 📄 License
This project is for educational and internship purposes.


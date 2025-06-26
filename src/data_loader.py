# src/data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

def load_local_dataset(path="data/imdb_small.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"review": "text", "sentiment": "label"})

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    # Convert to Hugging Face dataset format
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
        "test": Dataset.from_pandas(test_df.reset_index(drop=True))
    })
    return dataset

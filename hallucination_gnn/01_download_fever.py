from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

ds = load_dataset("copenlu/fever_gold_evidence")

train = ds["train"].select(range(100000))
val = ds["validation"].select(range(15000))

pd.DataFrame(train).to_csv("data/train_raw.csv", index=False)
pd.DataFrame(val).to_csv("data/val_raw.csv", index=False)

print("Saved: data/train_raw.csv and data/val_raw.csv")

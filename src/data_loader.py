import os
import urllib.request
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "cleveland.csv")

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        print("Downloading Cleveland Heart Disease dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print(f"Saved to {DATA_FILE}")
    else:
        print(f"Dataset found at {DATA_FILE}")


def load_data():
    download_data()
    df = pd.read_csv(DATA_FILE, header=None, names=COLUMNS, na_values="?")
    # Binarise target: 0 = no disease, 1 = disease (original values 1-4)
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    X = df.drop(columns=["target"])
    y = df["target"]
    print(f"Loaded {len(df)} samples | features: {X.shape[1]} | "
          f"positive rate: {y.mean():.2%}")
    return X, y

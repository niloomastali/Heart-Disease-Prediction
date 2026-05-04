import sys
import os

# Allow running as: python src/main.py  OR  python -m src.main
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.model_selection import train_test_split

from data_loader import load_data
from preprocessor import preprocess
from models import train_models
from evaluate import evaluate_models


def main():
    print("=" * 50)
    print("  HeartFuzz-ML Pipeline")
    print("=" * 50)

    print("\n[1/4] Loading data")
    X, y = load_data()

    print("\n[2/4] Splitting data  (80 / 20 stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    print("\n[3/4] Preprocessing  (median imputation + standard scaling)")
    X_train_p, X_test_p = preprocess(X_train, X_test)

    print("\n[4/4] Training & evaluating models")
    trained = train_models(X_train_p, y_train)
    metrics = evaluate_models(trained, X_test_p, y_test)

    print("\n" + "=" * 70)
    print(f"{'Model':<22} {'AUC':>7} {'F1':>7} {'Precision':>10} {'Recall':>8}")
    print("-" * 70)
    for name, m in metrics.items():
        print(
            f"{name:<22} {m['AUC']:>7.4f} {m['F1']:>7.4f} "
            f"{m['Precision']:>10.4f} {m['Recall']:>8.4f}"
        )
    print("=" * 70)
    print("\nDone.")


if __name__ == "__main__":
    main()

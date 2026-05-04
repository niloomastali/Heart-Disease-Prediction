import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.json")


def evaluate_models(trained_models, X_test, y_test, best_params=None, cv_scores=None):
    os.makedirs(PLOTS_DIR, exist_ok=True)

    metrics = {}
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

        metrics[name] = {
            "AUC": round(float(auc), 4),
            "F1": round(float(f1), 4),
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "CV_AUC": cv_scores.get(name) if cv_scores else None,
            "best_params": best_params.get(name, {}) if best_params else {},
        }

        # Add model curve to combined ROC plot
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

        # Individual confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["No Disease", "Disease"]
        )
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title(f"Confusion Matrix — {name}")
        fig_cm.tight_layout()
        safe_name = name.replace(" ", "_")
        fig_cm.savefig(os.path.join(PLOTS_DIR, f"cm_{safe_name}.png"), dpi=120)
        plt.close(fig_cm)

    # Finalise combined ROC figure
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=150)
    plt.close(fig)

    # Persist metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved  -> {METRICS_FILE}")
    print(f"Plots saved    -> {PLOTS_DIR}")
    return metrics

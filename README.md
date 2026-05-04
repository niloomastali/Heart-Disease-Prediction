# HeartFuzz-ML

Binary classification of heart disease using the UCI Cleveland Heart Disease dataset.

## Dataset

303 patients, 13 features, binary target (0 = no disease, 1 = disease).  
Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)

The dataset is downloaded automatically on first run to `data/cleveland.csv`.

## Models

| Model               | Notes                       |
|---------------------|-----------------------------|
| Logistic Regression | L2 regularisation           |
| Decision Tree       | Gini impurity               |
| KNN                 | k=5, Euclidean distance      |
| Naive Bayes         | Gaussian likelihood         |
| SVM                 | RBF kernel, probability=True|
| AdaBoost            | 100 estimators              |
| Random Forest       | 100 estimators              |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/main.py
```

## Output

| Path                          | Contents                          |
|-------------------------------|-----------------------------------|
| `results/metrics.json`        | AUC, F1, Precision, Recall per model |
| `results/plots/roc_curves.png`| Combined ROC curves               |
| `results/plots/cm_<Model>.png`| Confusion matrix per model        |

## Pipeline

1. **data_loader.py** — download & load dataset, binarise target  
2. **preprocessor.py** — median imputation, standard scaling  
3. **models.py** — define and train all classifiers  
4. **evaluate.py** — compute metrics, save plots and JSON  
5. **main.py** — orchestrate the full pipeline  

Stratified 80/20 train/test split, `random_state=42` throughout.

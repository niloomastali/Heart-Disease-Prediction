from sklearn.model_selection import GridSearchCV
from models import get_models

PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "liblinear"],
    },
    "Decision Tree": {
        "max_depth": [None, 3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
    },
    "Naive Bayes": {
        "var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3],
    },
    "SVM": {
        "C": [0.1, 1, 10],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.5, 1.0, 1.5],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    },
}


def tune_models(X_train, y_train, cv=5):
    tuned = {}
    best_params = {}
    cv_scores = {}
    for name, model in get_models().items():
        print(f"  Tuning {name}...")
        search = GridSearchCV(
            model,
            PARAM_GRIDS[name],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        best_params[name] = search.best_params_
        cv_scores[name] = round(search.best_score_, 4)
        print(f"    -> {search.best_params_}  |  CV AUC: {search.best_score_:.4f}")
    return tuned, best_params, cv_scores

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)


MODEL_CONFIGS = {
    "KNN": {
        "model": KNeighborsClassifier,
        "tagline": "You are the company you keep",
        "params": {
            "n_neighbors": [3, 7, 11],
            "weights": ["uniform", "distance"],
        },
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier,
        "tagline": "A series of yes/no questions",
        "params": {
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
    },
    "Random Forest": {
        "model": RandomForestClassifier,
        "tagline": "Wisdom of the crowd",
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10],
        },
    },
    "Naive Bayes": {
        "model": GaussianNB,
        "tagline": "Assume independence, predict fast",
        "params": {
            "var_smoothing": list(np.logspace(-10, -7, 5)),
        },
    },
    "SVM": {
        "model": SVC,
        "tagline": "Find the widest street between classes",
        "params": {
            "C": [0.1, 1.0, 10],
            "kernel": ["rbf", "linear"],
        },
    },
}


def train_model(model_name, X_train, y_train, X_test, y_test, random_state=42):
    """Train a single model with GridSearchCV and return results."""
    config = MODEL_CONFIGS[model_name]
    base_model = config["model"]

    # Add random_state where applicable
    model_kwargs = {}
    if model_name in ("Decision Tree", "Random Forest"):
        model_kwargs["random_state"] = random_state
    if model_name == "SVM":
        model_kwargs["random_state"] = random_state
        model_kwargs["probability"] = True

    estimator = base_model(**model_kwargs)

    grid = GridSearchCV(
        estimator, config["params"], cv=3, scoring="f1", n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model": best_model,
        "best_params": grid.best_params_,
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
        "tagline": config["tagline"],
    }


def train_all_models(X_train, y_train, X_test, y_test, random_state=42):
    """Train all 5 models and return results dict."""
    results = {}
    for name in MODEL_CONFIGS:
        results[name] = train_model(
            name, X_train, y_train, X_test, y_test, random_state
        )
    return results


def get_overfitting_data_dt(X_train, y_train, X_test, y_test,
                            max_depths=range(1, 31), random_state=42):
    """Train Decision Trees at various depths, return train/test accuracies."""
    train_accs, test_accs = [], []
    for d in max_depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=random_state)
        dt.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
        test_accs.append(accuracy_score(y_test, dt.predict(X_test)))
    return list(max_depths), train_accs, test_accs


def get_overfitting_data_knn(X_train, y_train, X_test, y_test,
                             k_values=range(1, 51)):
    """Train KNN at various K values, return train/test accuracies."""
    train_accs, test_accs = [], []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, knn.predict(X_train)))
        test_accs.append(accuracy_score(y_test, knn.predict(X_test)))
    return list(k_values), train_accs, test_accs


def get_overfitting_data_rf(X_train, y_train, X_test, y_test,
                            n_estimators_list=None, random_state=42):
    """Train Random Forest at various n_estimators."""
    if n_estimators_list is None:
        n_estimators_list = list(range(10, 501, 10))
    train_accs, test_accs = [], []
    for n in n_estimators_list:
        rf = RandomForestClassifier(
            n_estimators=n, max_depth=10, random_state=random_state, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, rf.predict(X_train)))
        test_accs.append(accuracy_score(y_test, rf.predict(X_test)))
    return n_estimators_list, train_accs, test_accs


def run_cross_validation(model_name, X, y, n_folds=5,
                         scoring="f1", random_state=42):
    """Run k-fold cross-validation for a given model."""
    config = MODEL_CONFIGS[model_name]

    model_kwargs = {}
    if model_name in ("Decision Tree", "Random Forest"):
        model_kwargs["random_state"] = random_state
    if model_name == "SVM":
        model_kwargs["random_state"] = random_state

    estimator = config["model"](**model_kwargs)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(estimator, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    return scores


def get_learning_curve_data(X, y, random_state=42):
    """Compute learning curves for Random Forest."""
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1
    )
    train_sizes, train_scores, val_scores = learning_curve(
        rf, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring="accuracy", n_jobs=-1,
        random_state=random_state,
    )
    return train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)

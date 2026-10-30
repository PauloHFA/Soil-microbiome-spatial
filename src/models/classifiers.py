"""
classifiers.py - Supervised classifier wrappers with class-weight balancing and fixed seeds.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score


def get_classifier(model_type: str = "rf", seed: int = 42, **kwargs) -> Any:
    """
    Instantiate supervised classifier with balanced class weights and fixed seed.

    Supported model_types:
        - 'rf': Random Forest
        - 'gbdt': Gradient Boosting Decision Tree
        - 'elasticnet': Logistic Regression with ElasticNet penalty
        - 'svm': Support Vector Classifier (RBF kernel with probability estimates)
    """
    m = model_type.lower()
    if m == "rf" or m == "random_forest":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 150),
            max_depth=kwargs.get("max_depth", None),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1
        )
    elif m == "gbdt" or m == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            learning_rate=kwargs.get("learning_rate", 0.1),
            random_state=seed
        )
    elif m == "elasticnet" or m == "logistic":
        return LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=kwargs.get("l1_ratio", 0.5),
            class_weight="balanced",
            max_iter=kwargs.get("max_iter", 1000),
            random_state=seed
        )
    elif m == "svm" or m == "svc":
        return SVC(
            kernel=kwargs.get("kernel", "rbf"),
            probability=True,
            class_weight="balanced",
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown classifier type '{model_type}'. Choose rf, gbdt, elasticnet, or svm.")


def train_classifier(clf: Any, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Fit classifier on training partition."""
    clf.fit(x_train, y_train)
    return clf


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute balanced accuracy and macro-F1 metrics."""
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)),
    }

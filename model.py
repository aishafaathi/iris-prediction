"""
Iris Species Prediction — KTU Machine Learning Theory Project

Compares KNN, Decision Tree, SVM and Random Forest.
Writes:
  model.pkl, scaler.pkl, model_comparison.csv
  results/model_comparison.png
  results/confusion_matrix.png
  results/feature_importance.png (for tree-based selected models)

Run: python model.py
"""

from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    feature_names = iris.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_grids = {
        "KNN": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7, 9],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [None, 2, 3, 4, 5],
                "model__min_samples_split": [2, 4, 6],
            },
        ),
        "SVM": (
            SVC(),
            {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 3, 5],
            },
        ),
    }

    rows = []
    trained = {}

    for name, (classifier, param_grid) in model_grids.items():
        print(f"Training and tuning {name}...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", classifier),
        ])
        search = GridSearchCV(
            pipeline,
            param_grid,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        predictions = search.predict(X_test)

        rows.append({
            "Model": name,
            "CV Accuracy": search.best_score_,
            "Test Accuracy": accuracy_score(y_test, predictions),
            "Precision (Macro)": precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "Recall (Macro)": recall_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "F1-score (Macro)": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "Best Parameters": str(search.best_params_),
        })
        trained[name] = {"search": search, "predictions": predictions}

    results = pd.DataFrame(rows).sort_values(
        ["CV Accuracy", "F1-score (Macro)"], ascending=False
    ).reset_index(drop=True)
    results.to_csv(BASE_DIR / "model_comparison.csv", index=False)

    # Select by cross-validation score; do not tune against the test set.
    selected_name = results.iloc[0]["Model"]
    selected = trained[selected_name]
    best_pipeline = selected["search"].best_estimator_
    selected_model = best_pipeline.named_steps["model"]
    selected_scaler = best_pipeline.named_steps["scaler"]
    predictions = selected["predictions"]

    # Keep compatibility with app.py: scaler.transform(inputs), then model.predict().
    with open(BASE_DIR / "model.pkl", "wb") as file:
        pickle.dump(selected_model, file)
    with open(BASE_DIR / "scaler.pkl", "wb") as file:
        pickle.dump(selected_scaler, file)

    # Model comparison chart.
    x = list(range(len(results)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar([i - width / 2 for i in x], results["CV Accuracy"], width,
           label="5-fold CV accuracy")
    ax.bar([i + width / 2 for i in x], results["Test Accuracy"], width,
           label="Test accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(results["Model"], rotation=12, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Iris Classifier Performance Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    # Confusion matrix for the selected classifier.
    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(cm, display_labels=target_names)
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {selected_name}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    # Feature importance is available for tree-based models.
    feature_plot = RESULTS_DIR / "feature_importance.png"
    if hasattr(selected_model, "feature_importances_"):
        importance = selected_model.feature_importances_
        order = importance.argsort()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([feature_names[i] for i in order], importance[order])
        ax.set_xlabel("Feature importance")
        ax.set_title(f"Feature Importance — {selected_name}")
        fig.tight_layout()
        fig.savefig(feature_plot, dpi=180)
        plt.close(fig)
    elif feature_plot.exists():
        feature_plot.unlink()

    print("\nMODEL COMPARISON")
    columns = [
        "Model", "CV Accuracy", "Test Accuracy",
        "Precision (Macro)", "Recall (Macro)", "F1-score (Macro)"
    ]
    print(results[columns].to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))
    print(f"\nSelected model (highest CV accuracy): {selected_name}")
    print("\nClassification report on held-out test set:")
    print(classification_report(
        y_test, predictions, target_names=target_names, zero_division=0
    ))
    print("\nSaved: model.pkl, scaler.pkl, model_comparison.csv")
    print("Saved: results/model_comparison.png and results/confusion_matrix.png")
    if feature_plot.exists():
        print("Saved: results/feature_importance.png")
    print("\nPlease test the Flask website locally before redeploying.")


if __name__ == "__main__":
    main()
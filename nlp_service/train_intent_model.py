"""
BMU Intent Classifier — Training & Evaluation Pipeline
=======================================================
Trains a TF-IDF + Logistic Regression intent classifier on the BMU dataset.

What this script produces:
  - intent_model.pkl         : Trained sklearn pipeline (TfidfVectorizer + LogisticRegression)
  - evaluation_report.txt    : Full classification report (Precision, Recall, F1, Accuracy)
  - confusion_matrix.png     : Visual confusion matrix heatmap
  - failure_cases.csv        : All test samples the model got wrong (for iterative improvement)

Run with:
    python train_intent_model.py
"""

import os
import sys
import json
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "backend", "data", "intent_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "intent_model.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.txt")
FAILURE_PATH = os.path.join(BASE_DIR, "failure_cases.csv")
MATRIX_PATH  = os.path.join(BASE_DIR, "confusion_matrix.png")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── 1. Load & Validate Dataset ───────────────────────────────────────────────
def load_data():
    logger.info(f"Loading dataset from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    # Basic validation
    assert "query" in df.columns and "intent" in df.columns, \
        "CSV must have 'query' and 'intent' columns"

    df = df.dropna(subset=["query", "intent"])
    df["query"] = df["query"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[df["query"].str.len() > 1]

    logger.info(f"  Total samples     : {len(df)}")
    logger.info(f"  Unique intents    : {df['intent'].nunique()}")
    logger.info(f"  Intent distribution:")
    for intent, count in df["intent"].value_counts().items():
        logger.info(f"    {intent:<20} {count} samples")

    return df


# ─── 2. Build ML Pipeline ─────────────────────────────────────────────────────
def build_pipeline():
    """
    TF-IDF vectorizes the text query into numerical features.
    Logistic Regression classifies it into one of the intent classes.
    Both steps are wrapped in a sklearn Pipeline for clean deployment.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),      # unigrams + bigrams for richer features
            max_features=8000,       # top 8000 vocabulary terms
            sublinear_tf=True,       # apply log normalization to TF
            strip_accents="unicode",
            analyzer="word",
            min_df=1,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,                   # regularization strength
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        ))
    ])
    return pipeline


# ─── 3. Train & Evaluate ──────────────────────────────────────────────────────
def train_and_evaluate(df):
    X = df["query"].values
    y = df["intent"].values
    class_names = sorted(df["intent"].unique())

    # ── Train/Test Split (80/20, stratified so each intent is represented) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"\nTrain samples : {len(X_train)}")
    logger.info(f"Test  samples : {len(X_test)}")

    # ── Build & Train ──────────────────────────────────────────────────────
    logger.info("\nTraining TF-IDF + Logistic Regression pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    # ── Test Set Evaluation ────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    logger.info(f"\n{'='*55}")
    logger.info(f"  TEST SET RESULTS")
    logger.info(f"{'='*55}")
    logger.info(f"  Accuracy        : {acc*100:.2f}%")
    logger.info(f"  Macro F1        : {macro_f1:.4f}")
    logger.info(f"  Weighted F1     : {weighted_f1:.4f}")

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    logger.info(f"\nPer-class Classification Report:\n{report}")

    # ── Cross-Validation (5-fold) for robustness proof ────────────────────
    logger.info("Running 5-fold Cross-Validation on full dataset...")
    cv_scores = cross_val_score(
        build_pipeline(), X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro",
        n_jobs=-1
    )
    logger.info(f"  CV F1 (macro) scores : {[round(s,4) for s in cv_scores]}")
    logger.info(f"  CV Mean ± Std        : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return pipeline, X_test, y_test, y_pred, y_proba, class_names, report, acc, macro_f1, cv_scores


# ─── 4. Confusion Matrix ──────────────────────────────────────────────────────
def save_confusion_matrix(y_test, y_pred, class_names):
    try:
        import matplotlib
        matplotlib.use("Agg")   # Non-interactive backend (no display needed)
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True, fmt="d",
            xticklabels=class_names, yticklabels=class_names,
            cmap="Blues", linewidths=0.5,
            ax=ax
        )
        ax.set_xlabel("Predicted Intent", fontsize=12)
        ax.set_ylabel("True Intent", fontsize=12)
        ax.set_title("BMU Intent Classifier — Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(MATRIX_PATH, dpi=150)
        plt.close()
        logger.info(f"  Confusion matrix saved → {MATRIX_PATH}")
    except ImportError:
        logger.warning("matplotlib/seaborn not installed — skipping confusion matrix plot.")
        logger.warning("Install with: pip install matplotlib seaborn")


# ─── 5. Failure Case Analysis ─────────────────────────────────────────────────
def save_failure_cases(X_test, y_test, y_pred, y_proba, pipeline):
    """Export every misclassified sample so we can analyze failure patterns."""
    failures = []
    classes = pipeline.classes_

    for i, (query, true, pred) in enumerate(zip(X_test, y_test, y_pred)):
        if true != pred:
            confidence = float(np.max(y_proba[i]))
            true_prob   = float(y_proba[i][list(classes).index(true)])
            failures.append({
                "query"          : query,
                "true_intent"    : true,
                "predicted_intent": pred,
                "confidence"     : round(confidence, 4),
                "true_intent_prob": round(true_prob, 4),
            })

    if failures:
        df_fail = pd.DataFrame(failures).sort_values("confidence", ascending=False)
        df_fail.to_csv(FAILURE_PATH, index=False)
        logger.info(f"  Failure cases ({len(failures)}) saved → {FAILURE_PATH}")
    else:
        logger.info("  No failure cases — 100% accuracy on test set!")

    return failures


# ─── 6. Save Evaluation Report ────────────────────────────────────────────────
def save_report(report, acc, macro_f1, cv_scores, num_failures, total_test):
    lines = [
        "BMU Intent Classifier — Evaluation Report",
        "=" * 55,
        f"Test Accuracy     : {acc*100:.2f}%",
        f"Macro F1 Score    : {macro_f1:.4f}",
        f"CV Mean F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        f"Failure Cases     : {num_failures} / {total_test}",
        "",
        "Per-class Classification Report:",
        report,
        "",
        "Model: TF-IDF (ngram 1-2, max_features=8000) + Logistic Regression (C=5.0)",
        "Dataset: backend/data/intent_dataset.csv",
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"  Full report saved → {REPORT_PATH}")


# ─── 7. Save Trained Model ────────────────────────────────────────────────────
def save_model(pipeline):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    size_kb = os.path.getsize(MODEL_PATH) / 1024
    logger.info(f"  Model saved → {MODEL_PATH} ({size_kb:.1f} KB)")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("\n" + "="*55)
    logger.info("  BMU INTENT CLASSIFIER — TRAINING PIPELINE")
    logger.info("="*55 + "\n")

    # 1. Load
    df = load_data()

    # 2. Train & Evaluate
    pipeline, X_test, y_test, y_pred, y_proba, class_names, report, acc, macro_f1, cv_scores = \
        train_and_evaluate(df)

    # 3. Confusion Matrix
    logger.info("\nGenerating confusion matrix...")
    save_confusion_matrix(y_test, y_pred, class_names)

    # 4. Failure Cases
    logger.info("Analyzing failure cases...")
    failures = save_failure_cases(X_test, y_test, y_pred, y_proba, pipeline)

    # 5. Save Report
    logger.info("Saving evaluation report...")
    save_report(report, acc, macro_f1, cv_scores, len(failures), len(X_test))

    # 6. Save Model
    logger.info("Saving trained model...")
    save_model(pipeline)

    logger.info("\n" + "="*55)
    logger.info("  DONE! Summary:")
    logger.info(f"  Accuracy  : {acc*100:.2f}%")
    logger.info(f"  Macro F1  : {macro_f1:.4f}")
    logger.info(f"  Failures  : {len(failures)}/{len(X_test)} test samples misclassified")
    logger.info("="*55 + "\n")

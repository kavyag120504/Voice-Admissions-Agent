# -*- coding: utf-8 -*-
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
"""
Child vs Adult Voice Classifier -- MESD Dataset
================================================
Uses the Mexican Emotional Speech Database (MESD) filename convention:
  Emotion_Gender_A_word.wav  -> Adult
  Emotion_Gender_B_word.wav  -> Child

Pipeline:
  1. Parse filenames to extract labels (A=adult, B=child)
  2. Extract acoustic features: MFCC + Delta-MFCC + Mel-spectrogram stats + Pitch
  3. Train SVM classifier (80/20 stratified split)
  4. Evaluate: Accuracy, Precision, Recall, F1, Confusion Matrix
  5. Save model as child_adult_model.pkl

Run:
    pip install librosa scikit-learn numpy pandas matplotlib seaborn soundfile
    python train_child_adult_classifier.py
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import soundfile as sf

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ---- Paths ----------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MESD_DIR   = os.path.join(BASE_DIR, "..", "..", "cy34mh68j9-5", "cy34mh68j9-5",
                           "Mexican Emotional Speech Database (MESD)")
MODEL_PATH = os.path.join(BASE_DIR, "child_adult_model.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "child_adult_report.txt")
MATRIX_PATH = os.path.join(BASE_DIR, "child_adult_confusion_matrix.png")
FEATURES_CACHE = os.path.join(BASE_DIR, "child_adult_features.npz")


# ---- 1. Parse Filenames & Build Label Map ------------------------------------
def parse_mesd_labels(mesd_dir):
    """
    MESD filename convention: Emotion_Gender_AgeGroup_word.wav
      AgeGroup = A -> adult
      AgeGroup = B -> child
    Returns list of (filepath, label) tuples.
    """
    samples = []
    if not os.path.exists(mesd_dir):
        print(f"ERROR: MESD directory not found at:\n  {mesd_dir}")
        sys.exit(1)

    for fname in os.listdir(mesd_dir):
        if not fname.lower().endswith(".wav"):
            continue
        parts = fname.split("_")
        # Need at least: Emotion_Gender_AgeGroup_word.wav -> 4 parts
        if len(parts) < 3:
            continue
        age_group = parts[2].upper()
        if age_group == "A":
            label = "adult"
        elif age_group == "B":
            label = "child"
        else:
            continue  # skip unknown

        fpath = os.path.join(mesd_dir, fname)
        samples.append((fpath, label))

    return samples


# ---- 2. Feature Extraction ---------------------------------------------------
def extract_features(fpath, sr=22050, n_mfcc=40):
    """
    Extract a fixed-length feature vector from a WAV file:
      - 40 MFCCs (mean + std)       -> 80 features
      - 40 Delta-MFCCs (mean + std) -> 80 features
      - Mel-spectrogram (mean + std) -> depends on n_mels=64 -> 128 features
      - Fundamental frequency / pitch (mean, std, min, max) -> 4 features
      - Zero-crossing rate (mean + std) -> 2 features
      - RMS energy (mean + std) -> 2 features
      - Spectral centroid (mean + std) -> 2 features
      Total: ~298 features
    """
    try:
        y, sr = librosa.load(fpath, sr=sr, mono=True, duration=5.0)
        if len(y) < sr * 0.1:   # skip files shorter than 0.1s
            return None

        feats = []

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        feats.extend(np.mean(mfcc, axis=1))
        feats.extend(np.std(mfcc, axis=1))

        # Delta-MFCCs (captures rate of change -> key for child vs adult)
        delta_mfcc = librosa.feature.delta(mfcc)
        feats.extend(np.mean(delta_mfcc, axis=1))
        feats.extend(np.std(delta_mfcc, axis=1))

        # Mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feats.extend(np.mean(mel_db, axis=1))
        feats.extend(np.std(mel_db, axis=1))

        # Fundamental frequency (pitch) -> children have higher F0
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([0.0])
        if len(f0_voiced) == 0:
            f0_voiced = np.array([0.0])
        feats.extend([
            np.mean(f0_voiced),
            np.std(f0_voiced),
            np.min(f0_voiced),
            np.max(f0_voiced),
        ])

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        feats.extend([np.mean(zcr), np.std(zcr)])

        # RMS energy
        rms = librosa.feature.rms(y=y)
        feats.extend([np.mean(rms), np.std(rms)])

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        feats.extend([np.mean(centroid), np.std(centroid)])

        return np.array(feats, dtype=np.float32)

    except Exception as e:
        print(f"  WARN: Failed to process {os.path.basename(fpath)}: {e}")
        return None


# ---- 3. Load or Build Feature Matrix -----------------------------------------
def load_features(samples):
    """Extract features from all audio files. Uses cache if available."""
    if os.path.exists(FEATURES_CACHE):
        print("  Loading cached features...")
        data = np.load(FEATURES_CACHE, allow_pickle=True)
        return data["X"], data["y"]

    print(f"  Extracting features from {len(samples)} audio files...")
    X, y = [], []
    errors = 0
    for i, (fpath, label) in enumerate(samples):
        if i % 50 == 0:
            print(f"    [{i}/{len(samples)}] processing...")
        feats = extract_features(fpath)
        if feats is not None:
            X.append(feats)
            y.append(label)
        else:
            errors += 1

    X = np.array(X)
    y = np.array(y)
    print(f"  Done. {len(X)} valid samples, {errors} skipped.")

    # Save cache for future runs
    np.savez(FEATURES_CACHE, X=X, y=y)
    print(f"  Features cached -> {FEATURES_CACHE}")
    return X, y


# ---- 4. Train & Evaluate -----------------------------------------------------
def train_and_evaluate(X, y):
    classes, counts = np.unique(y, return_counts=True)
    print(f"\n  Dataset:")
    for c, n in zip(classes, counts):
        print(f"    {c:<8} : {n} samples")

    # 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # SVM pipeline with StandardScaler
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=42
        ))
    ])

    print("\n  Training SVM (RBF kernel)...")
    pipeline.fit(X_train, y_train)
    print("  Training complete.")

    # Test set evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=sorted(classes), digits=4)

    print(f"\n{'='*52}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*52}")
    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Macro F1        : {macro_f1:.4f}")
    print(f"  Weighted F1     : {weighted_f1:.4f}")
    print(f"\n{report}")

    # 5-fold Cross-Validation
    print("  Running 5-fold Cross-Validation...")
    cv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42))
    ])
    cv_scores = cross_val_score(
        cv_pipeline, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_macro", n_jobs=-1
    )
    print(f"  CV F1 scores : {[round(s, 4) for s in cv_scores]}")
    print(f"  CV Mean±Std  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return pipeline, X_test, y_test, y_pred, classes, report, acc, macro_f1, cv_scores


# ---- 5. Confusion Matrix ------------------------------------------------------
def save_confusion_matrix(y_test, y_pred, classes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = confusion_matrix(y_test, y_pred, labels=sorted(classes))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=sorted(classes),
                    yticklabels=sorted(classes),
                    cmap="Blues", linewidths=0.5, ax=ax)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title("Child vs Adult Voice Classifier\nConfusion Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(MATRIX_PATH, dpi=150)
        plt.close()
        print(f"  Confusion matrix -> {MATRIX_PATH}")
    except ImportError:
        print("  WARN: matplotlib/seaborn not installed, skipping plot.")


# ---- 6. Save Report & Model --------------------------------------------------
def save_report(report, acc, macro_f1, cv_scores, total_test, total_train):
    lines = [
        "Child vs Adult Voice Classifier -- Evaluation Report",
        "=" * 52,
        f"Dataset         : MESD (Mexican Emotional Speech Database)",
        f"Labeling        : Filename _A_ = adult, _B_ = child",
        f"Features        : MFCC(40) + Delta-MFCC + Mel-spec(64) + Pitch + ZCR + RMS",
        f"Model           : SVM (RBF kernel, C=10, gamma=scale)",
        f"Train samples   : {total_train}",
        f"Test  samples   : {total_test}",
        "",
        f"Test Accuracy   : {acc*100:.2f}%",
        f"Macro F1        : {macro_f1:.4f}",
        f"CV Mean F1 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        "",
        "Per-class Classification Report:",
        report,
    ]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report -> {REPORT_PATH}")


# ---- Main --------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 52)
    print("  CHILD vs ADULT VOICE CLASSIFIER -- MESD")
    print("=" * 52 + "\n")

    # 1. Parse labels from filenames
    print("Step 1: Parsing MESD filenames...")
    samples = parse_mesd_labels(MESD_DIR)
    adult_count = sum(1 for _, l in samples if l == "adult")
    child_count = sum(1 for _, l in samples if l == "child")
    print(f"  Found {len(samples)} audio files -> adult={adult_count}, child={child_count}")

    if len(samples) < 20:
        print("ERROR: Too few samples found. Check MESD path.")
        sys.exit(1)

    # 2. Extract features
    print("\nStep 2: Extracting acoustic features (MFCC, pitch, mel-spectrogram)...")
    X, y = load_features(samples)

    # 3. Train & evaluate
    print("\nStep 3: Training SVM classifier...")
    pipeline, X_test, y_test, y_pred, classes, report, acc, macro_f1, cv_scores = \
        train_and_evaluate(X, y)

    # 4. Confusion matrix
    print("\nStep 4: Generating confusion matrix...")
    save_confusion_matrix(y_test, y_pred, classes)

    # 5. Save report
    print("\nStep 5: Saving evaluation report...")
    save_report(report, acc, macro_f1, cv_scores, len(X_test), len(X) - len(X_test))

    # 6. Save model
    print("\nStep 6: Saving model...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"  Model -> {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1024:.1f} KB)")

    print(f"\n{'='*52}")
    print(f"  DONE!")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Macro F1 : {macro_f1:.4f}")
    print(f"{'='*52}\n")

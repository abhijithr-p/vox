# train_ml.py
import os
import glob
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from data_utils import extract_features, extract_features_batch

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_dataset_from_dirs(base_dir, classes):
    paths = []
    labels = []
    for idx, cls in enumerate(classes):
        pattern = os.path.join(base_dir, cls, "*.wav")
        for p in glob.glob(pattern):
            paths.append(p)
            labels.append(idx)
    return paths, labels

def synthetic_demo_dataset(n_samples=200, n_features=13*2 + 5):
    # Create synthetic features for quick demo/training (not real audio)
    rng = np.random.RandomState(42)
    X = []
    y = []
    for i in range(n_samples):
        if i % 3 == 0:
            # fluent
            base = rng.normal(loc=0.0, scale=1.0, size=n_features)
            label = 0
        elif i % 3 == 1:
            # stutter-like
            base = rng.normal(loc=0.5, scale=1.2, size=n_features)
            label = 1
        else:
            # dysarthria-like
            base = rng.normal(loc=-0.3, scale=1.5, size=n_features)
            label = 2
        X.append(base)
        y.append(label)
    return np.vstack(X), np.array(y)

def main(args):
    classes = args.classes.split(",")  # e.g. "fluent,stutter,dysarthria"
    if args.demo:
        print("Building synthetic demo dataset...")
        # compute feature size by extracting on an empty generated audio (we'll approximate)
        n_features = 13*2 + 5
        X, y = synthetic_demo_dataset(n_samples=300, n_features=n_features)
    else:
        print("Building dataset from directories...")
        paths, labels = build_dataset_from_dirs(args.data_dir, classes)
        if not paths:
            raise RuntimeError("No audio files found. Use --demo to run a synthetic demo or provide data.")
        print(f"Found {len(paths)} files.")
        X = extract_features_batch(paths)
        y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.18, random_state=42, stratify=y)
    print("Training RandomForest...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_val)
    print(classification_report(y_val, y_pred, target_names=classes))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save model and classes
    joblib.dump(clf, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(classes, os.path.join(MODEL_DIR, "classes.joblib"))
    print("Model and class mapping saved to", MODEL_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Base data dir with subfolders per class")
    parser.add_argument("--classes", type=str, default="fluent,stutter,dysarthria", help="comma-separated class names")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo dataset (no real audio needed)")
    args = parser.parse_args()
    main(args)

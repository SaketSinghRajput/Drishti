#!/usr/bin/env python3
"""Fixed Accident Prediction Demo"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import xgboost as xgb
from tqdm import tqdm

# PYTHONPATH: ensure repository root (parent of `brain`) is on sys.path so
# `import brain.*` resolves when running the script from inside `brain/`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from brain.feature_engineering.extractor import FeatureExtractor
from brain.config.config_loader import ConfigLoader

cfg_path = 'config_examples/dev_sim.yaml'
cfg = ConfigLoader(cfg_path)
extractor = FeatureExtractor(cfg)

def generate_fake_data(n_samples=100):
    """Generate proper sensor windows (lists of readings). Return a list of dicts."""
    np.random.seed(42)
    data = []
    for i in tqdm(range(n_samples), desc="Generating fake data"):
        # Proper window format: lists of readings
        lbl = int(np.random.choice([0, 1], p=[0.9, 0.1]))
        window = {
            'window_id': f'win_{i}',
            'trip_id': f'trip_{i//10}',
            'timestamp': '2024-01-01T00:00:00Z',
            'gps': [{'speed_kph': float(np.random.uniform(20, 80)), 'heading': float(np.random.uniform(0, 360))}],
            'imu': [{'yaw_rate': float(np.random.normal(0, 5)), 'ax': float(np.random.normal(0, 2)), 'ay': float(np.random.normal(0, 1)), 'az': 9.8}],
            'vehicle_controls': [{'steering_angle': float(np.random.normal(0, 20)), 'brake_pressure': float(np.random.uniform(0, 1)), 'throttle_position': float(np.random.uniform(0, 1))}],
            'driver_fatigue': [{'drowsiness_score': float(np.random.uniform(0, 1)), 'perclos': float(np.random.uniform(0, 0.5))}],
            'label': lbl,
            'labels': {'label': lbl},
        }
        data.append(window)
    return data

def main():
    print("🚗 Fixed Accident Prediction Demo")
    
    # 1. Generate fake data
    print("1. Generating fake data...")
    df_raw = generate_fake_data(100)
    # Save a CSV snapshot for inspection
    pd.DataFrame(df_raw).to_csv('data/demo_accidents_raw.csv', index=False)
    
    # 2. Extract features
    print("2. Extracting features...")
    df_windows = df_raw  # already a list of window dicts
    df_features = extractor.extract_to_dataframe(df_windows)
    # Ensure it's a DataFrame
    if not isinstance(df_features, pd.DataFrame):
        df_features = pd.DataFrame(df_features)

    # Ensure 'label' column exists (extractor emits label_{k} entries from window['labels'])
    if 'label' not in df_features.columns:
        labels_list = []
        for w in df_windows:
            if isinstance(w, dict):
                if 'label' in w:
                    labels_list.append(int(w.get('label')))
                else:
                    lab = None
                    if isinstance(w.get('labels'), dict):
                        lab = w.get('labels').get('label')
                    labels_list.append(int(lab) if lab is not None else 0)
            else:
                labels_list.append(0)
        df_features['label'] = labels_list

    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    df_features.to_csv('data/demo_features.csv', index=False)
    print(f"Features shape: {df_features.shape}")
    
    # 3. Train/Test split
    X = df_features.select_dtypes(include=[np.number]).dropna(axis=1)  # Numeric features
    y = df_features['label'].fillna(0).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    
    # 4. Train XGBoost
    print("3. Training XGBoost...")
    model = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/demo_risk_model.joblib')
    
    # 5. Predict + Accuracy
    print("4. Predicting on 10 test samples...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    for i in range(min(10, len(X_test))):
        print(f"Sample {i+1}: Risk={y_pred_proba[i]:.3f} ({'HIGH' if y_pred[i] else 'LOW'}) | True: {y_test.iloc[i]}")
    
    # Metrics
    roc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    print(f"ROC-AUC: {roc:.3f} ({roc*100:.1f}%)")
    print(f"Accuracy: {acc:.3f} ({acc*100:.1f}%)")
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    
    print("✅ Fixed! Model saved: models/demo_risk_model.joblib")

if __name__ == "__main__":
    main()
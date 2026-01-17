
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib


CSV_FILE = 'synthetic_telematics.csv' 
MODEL_FILENAME = 'telematics_xgb_model.joblib'
FEATURES_FILENAME = 'model_features.joblib'

def train_and_save_model():
    print(f"Loading data from {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found. Ensure synthetic_telematics.csv is in the project folder.")
        return

    TARGET = 'failure_within_30min'
    FEATURES = [
        'speed_kmh', 'acceleration', 'lat', 'lon', 'rpm', 
        'engine_temp_c', 'vibration', 'fuel_level_pct', 
        'engine_load', 'brake_pressure', 'battery_v', 'co2'
    ]

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training XGBoost Classifier...")
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])

    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )

    
    model.fit(X_train, y_train) 
    print("Training complete.")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nModel Performance (ROC AUC): {auc_score:.4f}")

    
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(FEATURES, FEATURES_FILENAME)

    print(f"\nModel saved to: {MODEL_FILENAME}")
    print(f"Features saved to: {FEATURES_FILENAME}")

if __name__ == '__main__':
    train_and_save_model()
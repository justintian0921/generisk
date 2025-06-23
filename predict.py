import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# === Load Model, Scaler, and Feature Names ===
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")
print(f"Loaded model with {len(feature_names)} features")

# === Load First 5 Rows from Test Data ===
input_df = pd.read_csv("test.csv").head(7)

# === Drop Unused Columns ===
drop_cols = ['Chr', 'Start', 'End', 'Ref', 'Alt', 'CLASS', 'gene', 'Binary_Class']
X_input = input_df.drop(columns=[col for col in drop_cols if col in input_df.columns], errors='ignore')

# === Align Features with Training Data ===
X_input = X_input.reindex(columns=feature_names, fill_value=0)

print(f"Input data shape after alignment: {X_input.shape}")
print(f"Features aligned: {list(X_input.columns) == feature_names}")

# === Fill Missing Values ===
X_input.fillna(X_input.median(), inplace=True)

# === Normalize Features ===
X_input_scaled = pd.DataFrame(scaler.transform(X_input), columns=X_input.columns)

# === Get Predictions and Probabilities ===
predictions = model.predict(X_input_scaled)
probabilities = model.predict_proba(X_input_scaled)

# === Output Predictions with Confidence ===
print("\n" + "="*60)
print("PREDICTION RESULTS WITH CONFIDENCE")
print("="*60)

for i in range(len(predictions)):
    pred = predictions[i]
    prob = probabilities[i]
    
    # Get confidence (probability of predicted class)
    confidence = prob[pred] * 100
    
    label = "Pathogenic" if pred == 1 else "Benign"
    print(f"\nSample {i + 1}: {label} ({confidence:.1f}%)")
    print(f"  Probabilities: Benign={prob[0]*100:.1f}%, Pathogenic={prob[1]*100:.1f}%")
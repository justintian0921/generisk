import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# === Load Train and Test Data ===
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# === Label Mapping ===
train_df['Binary_Class'] = train_df['CLASS'].map({1: 1, -1: 0})
test_df['Binary_Class'] = test_df['CLASS'].map({1: 1, -1: 0})

# === Feature Selection ===
drop_cols = ['Chr', 'Start', 'End', 'Ref', 'Alt', 'CLASS', 'gene']
X_train = train_df.drop(columns=drop_cols + ['Binary_Class'], errors='ignore')
y_train = train_df['Binary_Class']

X_test = test_df.drop(columns=drop_cols + ['Binary_Class'], errors='ignore')
y_test = test_df['Binary_Class']

# === Align Columns Between Train and Test ===
# Keep original order from X_train (more deterministic than using set())
common_features = [col for col in X_train.columns if col in X_test.columns]
X_train = X_train[common_features].copy()
X_test = X_test[common_features].copy()

# === Handle Missing Values ===
X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_train.median(), inplace=True)

# === Normalize Features and Keep Column Names ===
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# === Save Scaler and Feature Names ===
joblib.dump(scaler, "scaler.pkl")
# Save the exact feature names used during training
feature_names = X_train_scaled.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")
print(f"Saved {len(feature_names)} feature names to feature_names.pkl")

# === Train XGBoost Classifier ===
model = XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False, eval_metric="logloss")
model.fit(X_train_scaled, y_train)

# === Save Model ===
joblib.dump(model, "xgboost_model.pkl")

# === Evaluate on Test Set ===
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nAccuracy: {accuracy:.4f}")

# === Plot Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Harmful", "Harmful"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# === Plot Feature Importance ===
xgb_importance = model.feature_importances_
features = X_train.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": xgb_importance})
importance_df.sort_values(by="Importance", ascending=False, inplace=True)

plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(15), palette="viridis")
plt.title("Top 15 Feature Importances")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining complete. Model, scaler, feature names, and graphs saved.")
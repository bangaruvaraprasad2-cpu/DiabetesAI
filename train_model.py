import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

print("=" * 60)
print("  DIABETES PREDICTION MODEL TRAINER")
print("=" * 60)

# ─── Load Dataset ───────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), 'data', 'diabetes.csv')
df = pd.read_csv(data_path)
print(f"\n✔  Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ─── Feature Engineering ────────────────────────────────────────
# Replace zero values in medical columns with median (they indicate missing data)
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    median_val = df[col][df[col] != 0].median()
    df[col] = df[col].replace(0, median_val)

print(f"✔  Zero-value imputation done for: {', '.join(zero_cols)}")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# ─── Train/Test Split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"✔  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ─── Scaling ────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── Model Training ─────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)
print("✔  Random Forest model trained (200 trees)")

# ─── Evaluation ─────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n  Accuracy : {acc * 100:.2f}%")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# ─── Feature Importance ─────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\n  Feature Importances:")
for feat, imp in importances.sort_values(ascending=False).items():
    bar = "█" * int(imp * 40)
    print(f"    {feat:<30} {bar} {imp:.4f}")

# ─── Save Model & Scaler ────────────────────────────────────────
os.makedirs(os.path.join(os.path.dirname(__file__), 'model'), exist_ok=True)
joblib.dump(model,  os.path.join(os.path.dirname(__file__), 'model', 'model.pkl'))
joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl'))

print("\n✔  model.pkl  saved → model/model.pkl")
print("✔  scaler.pkl saved → model/scaler.pkl")
print("\n" + "=" * 60)
print("  TRAINING COMPLETE! ✅")
print("=" * 60 + "\n")

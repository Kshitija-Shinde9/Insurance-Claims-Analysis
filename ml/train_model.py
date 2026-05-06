"""
train_model.py
==============
Run this ONCE to train the Random Forest model and save all artifacts.
After running, the Streamlit dashboard loads these files instead of
retraining every time.

Usage:
    python train_model.py

Output files created:
    rf_model.pkl         - trained Random Forest classifier
    encoders.pkl         - LabelEncoders for categorical columns
    features.pkl         - ordered feature list
    feature_importance.csv - feature importance rankings
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "../data/processed/fremtpl2_policy_claims.csv"
OUTPUT_DIR   = "."   # saves artifacts next to this script

# ── Step 1: Load data ─────────────────────────────────────────────────────────
print("=" * 55)
print("  French Motor Insurance — ML Training Pipeline")
print("=" * 55)
print(f"\n[1/5] Loading data from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"      Loaded {len(df):,} rows × {len(df.columns)} columns")

# ── Step 2: Derive Risk Profile target ───────────────────────────────────────
print("\n[2/5] Deriving Risk Profile target from BonusMalus + claims...")

def assign_risk(row):
    """
    Risk Profile logic:
      0 = Low       : BonusMalus ≤ 60  AND no claims
      1 = Medium    : BonusMalus ≤ 80  OR  small claim (<€1,000)
      2 = High      : BonusMalus ≤ 120 OR  moderate claim (<€5,000)
      3 = Very High : BonusMalus > 120  OR  large claim (≥€5,000)
    """
    bm = row['BonusMalus']
    hc = row['HasClaim']
    ca = row['TotalClaimAmount']
    if   bm <= 60  and not hc:           return 0
    elif bm <= 80  or (hc and ca < 1000):  return 1
    elif bm <= 120 or (hc and ca < 5000):  return 2
    else:                                  return 3

df['Risk_Profile'] = df.apply(assign_risk, axis=1)
df['Risk_Label']   = df['Risk_Profile'].map({
    0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'
})

dist = df['Risk_Label'].value_counts()
print(f"      Distribution:")
for label in ['Low','Medium','High','Very High']:
    n = dist.get(label,0)
    print(f"        {label:12s}: {n:>7,} ({n/len(df)*100:.1f}%)")

# ── Step 3: Feature engineering ───────────────────────────────────────────────
print("\n[3/5] Encoding features...")

# Only real underwriting inputs — NO data leakage
# PurePremium and Frequency are EXCLUDED because they are
# derived from claims (the same data that defines the target)
FEATURES = [
    'VehPower',    # vehicle engine power rating
    'VehAge',      # vehicle age in years
    'DrivAge',     # driver age in years
    'BonusMalus',  # core actuarial risk score (50=best, 230=worst)
    'Density',     # population density of driver area
    'Exposure',    # fraction of year policy was active
    'VehBrand',    # vehicle brand (categorical)
    'VehGas',      # fuel type: Regular or Diesel (categorical)
    'Area',        # area code A–F: rural to dense urban (categorical)
    'Region',      # French region (categorical)
]

CAT_COLS = ['VehBrand', 'VehGas', 'Area', 'Region']

df_model = df[FEATURES + ['Risk_Profile']].dropna().copy()
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le
    print(f"      Encoded '{col}': {list(le.classes_)}")

X = df_model[FEATURES]
y = df_model['Risk_Profile']

# ── Step 4: Train model ───────────────────────────────────────────────────────
print(f"\n[4/5] Training Random Forest on {len(X):,} samples...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)

print(f"\n       Accuracy: {acc*100:.2f}%")
print()
print("      Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Low','Medium','High','Very High'],
    digits=3
))

print("      Confusion Matrix:")
print(f"      {'':14s}  {'Low':>7s}  {'Medium':>7s}  {'High':>7s}  {'VeryHigh':>8s}")
labels = ['Low','Medium','High','Very High']
for i, row_label in enumerate(labels):
    row_str = "  ".join([f"{cm[i,j]:>7,}" for j in range(4)])
    print(f"      {row_label:14s}  {row_str}")

fi = pd.DataFrame({
    'Feature':    FEATURES,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)
fi['Importance_pct'] = (fi['Importance'] * 100).round(2)

print()
print("      Feature Importances:")
for _, row in fi.iterrows():
    bar = "█" * int(row['Importance_pct'] * 1.5)
    print(f"      {row['Feature']:15s}  {row['Importance_pct']:5.1f}%  {bar}")

# ── Step 5: Save artifacts ────────────────────────────────────────────────────
print(f"\n[5/5] Saving model artifacts to '{OUTPUT_DIR}/'...")

with open(os.path.join(OUTPUT_DIR, 'rf_model.pkl'),   'wb') as f: pickle.dump(rf,       f)
with open(os.path.join(OUTPUT_DIR, 'encoders.pkl'),   'wb') as f: pickle.dump(encoders, f)
with open(os.path.join(OUTPUT_DIR, 'features.pkl'),   'wb') as f: pickle.dump(FEATURES, f)
fi.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)

print("       rf_model.pkl          — trained Random Forest")
print("       encoders.pkl          — LabelEncoders for categorical columns")
print("       features.pkl          — ordered feature list")
print("       feature_importance.csv — feature importances")
print()
print("=" * 55)
print("  Training complete! Now run:")
print("  streamlit run app_merged_dashboard.py")
print("=" * 55)

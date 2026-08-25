import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

print("=" * 50)
print("💳 Credit Card Default - Model Training")
print("=" * 50)

# 1. Load Data
print("\n📊 Loading data...")
df = pd.read_csv('data/raw/CreditCard_5.csv')  # ✅ Fixed path

# Column names set karo
if df.shape[1] == 24:
    df.columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 
                 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default']

print(f"✅ Data loaded: {len(df)} rows")
print(f"   Default rate: {df['default'].mean():.2%}")

# 2. Features
print("\n🔧 Preparing features...")
features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
            'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'PAY_0', 'PAY_2',
            'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
cat_features = ['SEX', 'EDUCATION', 'MARRIAGE']

X = df[features + cat_features].copy()
y = df['default']

for col in cat_features:
    X[col] = X[col].astype('category').cat.codes

print(f"✅ Features: {X.shape[1]}")

# 3. Split
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

# 4. Scale
print("\n📏 Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Done")

# 5. Train
print("\n🤖 Training model...")
model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
print("✅ Model trained")

# 6. Evaluate
print("\n📊 Evaluating...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("=" * 50)
print("📈 Model Performance:")
print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"   Precision: {precision_score(y_test, y_pred):.4f}")
print(f"   Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"   F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"   ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
print("=" * 50)

# 7. Save
print("\n💾 Saving model...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/logistic_regression.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')

with open('models/threshold.txt', 'w') as f:
    f.write('0.5')

print("✅ Model saved in models/ folder")
print("\n🎉 Training complete!")
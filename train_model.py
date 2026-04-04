"""
Cardeko Car Price Prediction — Model Training
Uses real Cardekho dataset (15,411 cars)
Algorithm: RandomForestRegressor
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

CSV_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cardekho_dataset.csv')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"   Raw shape: {df.shape}")

df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
df['brand'] = df['brand'].str.strip().str.title()

# Drop erroneous rows
df = df[df['seats'] > 0]
upper = df['selling_price'].quantile(0.995)
df = df[df['selling_price'] <= upper]
df = df[(df['mileage'] >= 4) & (df['mileage'] <= 40)]

print(f"   Clean shape: {df.shape}")
print(f"   Price range: Rs.{df['selling_price'].min():,} - Rs.{df['selling_price'].max():,}")

# ─────────────────────────────────────────────
# 2. Feature Engineering
# ─────────────────────────────────────────────
df['km_per_year']       = df['km_driven'] / (df['vehicle_age'] + 1)
df['power_to_engine']   = df['max_power'] / df['engine']
df['mileage_per_power'] = df['mileage'] / (df['max_power'] + 1)
df['price_segment']     = pd.cut(
    df['engine'],
    bins=[0, 1000, 1500, 2000, 3000, 10000],
    labels=[0, 1, 2, 3, 4]
).astype(int)

# ─────────────────────────────────────────────
# 3. Encode Categoricals
# ─────────────────────────────────────────────
CAT_COLS = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type']
label_encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ─────────────────────────────────────────────
# 4. Train/Test Split
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'brand_enc', 'model_enc',
    'vehicle_age', 'km_driven',
    'seller_type_enc', 'fuel_type_enc', 'transmission_type_enc',
    'mileage', 'engine', 'max_power', 'seats',
    'km_per_year', 'power_to_engine', 'mileage_per_power', 'price_segment'
]
TARGET_COL = 'selling_price'

X = df[FEATURE_COLS]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 5. Train Random Forest
# ─────────────────────────────────────────────
print("\n Training RandomForestRegressor...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("Training complete!")

# ─────────────────────────────────────────────
# 6. Evaluate
# ─────────────────────────────────────────────
y_pred = rf.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
cv   = cross_val_score(rf, X, y, cv=5, scoring='r2', n_jobs=-1)

print(f"\nModel Performance:")
print(f"  MAE  : Rs.{mae:,.0f}  ({mae/100000:.2f} Lakhs)")
print(f"  RMSE : Rs.{rmse:,.0f}  ({rmse/100000:.2f} Lakhs)")
print(f"  R2   : {r2:.4f}  ({r2*100:.1f}%)")
print(f"  CV R2: {cv.mean():.4f} +/- {cv.std():.4f}")

print("\nTop Feature Importances:")
fi = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
for feat, imp in fi.head(10).items():
    bar = '#' * int(imp * 50)
    print(f"  {feat:<25} {bar}  {imp:.4f}")

# ─────────────────────────────────────────────
# 7. Save Artifacts
# ─────────────────────────────────────────────
brand_model_map = (
    df.groupby('brand')['model']
    .apply(lambda x: sorted(x.unique().tolist()))
    .to_dict()
)

artifacts = {
    'model':           rf,
    'label_encoders':  label_encoders,
    'feature_cols':    FEATURE_COLS,
    'cat_cols':        CAT_COLS,
    'brands':          sorted(df['brand'].unique().tolist()),
    'brand_model_map': brand_model_map,
    'fuel_types':      sorted(df['fuel_type'].unique().tolist()),
    'seller_types':    sorted(df['seller_type'].unique().tolist()),
    'transmissions':   sorted(df['transmission_type'].unique().tolist()),
    'metrics': {
        'mae':          round(mae, 0),
        'mae_lakhs':    round(mae / 100000, 2),
        'rmse':         round(rmse, 0),
        'rmse_lakhs':   round(rmse / 100000, 2),
        'r2':           round(r2, 4),
        'cv_r2_mean':   round(cv.mean(), 4),
        'cv_r2_std':    round(cv.std(), 4),
        'train_samples': int(X_train.shape[0]),
        'test_samples':  int(X_test.shape[0]),
        'total_samples': int(df.shape[0]),
    }
}

os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, 'cardeko_rf_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(artifacts, f)

print(f"\nModel saved to {model_path}")
print(f"File size: {os.path.getsize(model_path)/1024/1024:.1f} MB")
print("\nRun `python app.py` to start the Flask API!")

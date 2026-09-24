# =============================================================================
# Retail Customer Purchase Prediction — Training Script
# Converted from: Retail_Customer_Purchase_Prediction.ipynb
# Trains Logistic Regression + Random Forest, saves .pkl files to models/
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv("data/Retail_Customer_Purchase_Prediction.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. REMOVE DUPLICATES
# ---------------------------------------------------------------------------
df = df.drop_duplicates()
print(f"After removing duplicates: {df.shape[0]} rows")

# ---------------------------------------------------------------------------
# 3. FILL MISSING VALUES — CATEGORICAL (mode)
# ---------------------------------------------------------------------------
for col in ['Cart_Activity', 'Discount_Viewed', 'Discount_Used',
            'Device_Type', 'Traffic_Source']:
    df[col] = df[col].fillna(df[col].mode()[0])

# ---------------------------------------------------------------------------
# 4. FILL MISSING VALUES — NUMERICAL (median)
# ---------------------------------------------------------------------------
num_cols = [
    'Pages_Visited', 'Time_Spent', 'Previous_Visits', 'Products_Viewed',
    'Cart_Items', 'Wishlist_Items', 'Searches_Performed',
    'Product_Detail_Views', 'Previous_Purchases', 'Previous_Total_Spend',
    'Average_Order_Value', 'Days_Since_Last_Visit', 'Session_Count',
]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# ---------------------------------------------------------------------------
# 5. ENGINEER: Previous_Purchase_Missing (BEFORE filling Days_Since_Last_Purchase)
# ---------------------------------------------------------------------------
df['Previous_Purchase_Missing'] = df['Days_Since_Last_Purchase'].isnull().astype(int)

# Fill Days_Since_Last_Purchase with median
df['Days_Since_Last_Purchase'] = df['Days_Since_Last_Purchase'].fillna(
    df['Days_Since_Last_Purchase'].median()
)

# ---------------------------------------------------------------------------
# 6. DROP Customer_ID
# ---------------------------------------------------------------------------
df = df.drop('Customer_ID', axis=1)

# ---------------------------------------------------------------------------
# 7. ENGINEER: Engagement_Score and Has_Cart
# ---------------------------------------------------------------------------
df['Engagement_Score'] = (
    df['Pages_Visited'] +
    df['Products_Viewed'] +
    df['Product_Detail_Views'] +
    df['Searches_Performed']
)

df['Has_Cart'] = (df['Cart_Items'] > 0).astype(int)

# ---------------------------------------------------------------------------
# 8. SPLIT FEATURES / TARGET
# ---------------------------------------------------------------------------
x = df.drop('Purchase', axis=1)
y = df['Purchase']

# ---------------------------------------------------------------------------
# 9. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print(f"Train: {x_train.shape}  |  Test: {x_test.shape}")

# ---------------------------------------------------------------------------
# 10. BUILD PREPROCESSOR
# ---------------------------------------------------------------------------
categorical_cols = x_train.select_dtypes(include='object').columns
numerical_cols   = x_train.select_dtypes(include=np.number).columns

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ('num', num_transformer, numerical_cols),
    ('cat', cat_transformer, categorical_cols),
])

preprocessor.fit(x_train)

x_train_transformed = preprocessor.transform(x_train)
x_test_transformed  = preprocessor.transform(x_test)

# ---------------------------------------------------------------------------
# 11. LOGISTIC REGRESSION
# ---------------------------------------------------------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train_transformed, y_train)

y_pred_lr = log_model.predict(x_test_transformed)
y_prob_lr = log_model.predict_proba(x_test_transformed)[:, 1]

print("\n=== Logistic Regression ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob_lr):.4f}")

# ---------------------------------------------------------------------------
# 12. RANDOM FOREST
# ---------------------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(x_train_transformed, y_train)

rf_pred = rf_model.predict(x_test_transformed)
rf_prob = rf_model.predict_proba(x_test_transformed)[:, 1]

print("\n=== Random Forest ===")
print(f"Accuracy : {accuracy_score(y_test, rf_pred):.4f}")
print(f"Precision: {precision_score(y_test, rf_pred):.4f}")
print(f"Recall   : {recall_score(y_test, rf_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, rf_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, rf_prob):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=['No Purchase', 'Purchase']))

# ---------------------------------------------------------------------------
# 13. MODEL COMPARISON
# ---------------------------------------------------------------------------
comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'Logistic Regression': [
        accuracy_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_lr),
        roc_auc_score(y_test, y_prob_lr),
    ],
    'Random Forest': [
        accuracy_score(y_test, rf_pred),
        precision_score(y_test, rf_pred),
        recall_score(y_test, rf_pred),
        f1_score(y_test, rf_pred),
        roc_auc_score(y_test, rf_prob),
    ],
})
print("\n=== Model Comparison ===")
print(comparison.to_string(index=False))

# ---------------------------------------------------------------------------
# 14. PURCHASE LIKELIHOOD — example on first test customer
# ---------------------------------------------------------------------------
def purchase_indicator(prob):
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    else:
        return "Low"

new_customer = x_test.iloc[[0]]
probability  = rf_model.predict_proba(preprocessor.transform(new_customer))[:, 1][0]
print(f"\nExample — first test customer:")
print(f"Purchase Probability: {round(probability, 4)}")
print(f"Likelihood Indicator: {purchase_indicator(probability)}")

# ---------------------------------------------------------------------------
# 15. SAVE MODELS
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(log_model,   "models/logistic_regression.pkl")
joblib.dump(rf_model,    "models/rf_model.pkl")
joblib.dump(preprocessor,"models/preprocessor.pkl")

print("\nModels saved to models/")
print("  models/logistic_regression.pkl")
print("  models/rf_model.pkl")
print("  models/preprocessor.pkl")

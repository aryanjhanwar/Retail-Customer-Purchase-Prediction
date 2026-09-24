# Retail Customer Purchase Prediction

**PS-02 - Hackathon Project**
Predicts whether an online retail customer is likely to make a purchase based on their website interaction, browsing behaviour, and purchase history, powered by a Random Forest classifier.

---

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## Problem Statement

Online retailers lose significant revenue when potential customers leave without purchasing. This project builds a machine learning system that:

- Analyses real-time customer behaviour on the website
- Predicts **Purchase / No Purchase** with a probability score
- Classifies the likelihood as **High / Medium / Low**
- Helps retailers take proactive action (personalised offers, retargeting)

---

## Machine Learning Pipeline

### Dataset
- **Source:** Retail Customer Purchase Prediction Dataset
- **Size:** 14,561 rows x 21 features
- **Target:** `Purchase` (binary: 0 = No Purchase, 1 = Purchase)

### Features Used

| Category | Features |
|----------|----------|
| Browsing Behaviour | Pages Visited, Time Spent, Products Viewed, Product Detail Views, Searches Performed |
| Cart and Wishlist | Cart Items, Cart Activity, Wishlist Items |
| Discounts | Discount Viewed, Discount Used |
| Purchase History | Previous Purchases, Previous Total Spend, Average Order Value, Days Since Last Purchase |
| Session Info | Previous Visits, Days Since Last Visit, Session Count, Device Type, Traffic Source |
| Engineered Features | Engagement_Score, Has_Cart, Previous_Purchase_Missing |

### Feature Engineering

```python
Engagement_Score          = Pages_Visited + Products_Viewed + Product_Detail_Views + Searches_Performed
Has_Cart                  = 1 if Cart_Items > 0 else 0
Previous_Purchase_Missing = 1 if Days_Since_Last_Purchase was null else 0
```

### Preprocessing
- Categorical missing values filled with **mode**
- Numerical missing values filled with **median**
- `StandardScaler` for numerical features
- `OneHotEncoder` for categorical features
- `ColumnTransformer` to combine both

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 69.50% | 72.09% | 90.23% | 80.15% | 68.28% |
| **Random Forest (Selected)** | **68.67%** | **72.27%** | **87.72%** | **79.25%** | **67.70%** |

Random Forest was selected as the final model for deployment.

---

## Streamlit App

### Purchase Likelihood Thresholds

| Probability | Likelihood |
|-------------|-----------|
| >= 70% | HIGH |
| 40% - 69% | MEDIUM |
| < 40% | LOW |

### Input Sections
1. **Customer Behaviour** - pages, products, searches, time spent
2. **Purchase History** - past orders, spend, order value, days since last purchase
3. **Website Interaction** - cart items, discounts, session count
4. **Device / Traffic Information** - device type, traffic source

---

## Project Structure

```
retail-customer-purchase-prediction/
|
|-- app.py                    # Streamlit web application
|-- train.py                  # ML training script (reproduces models from scratch)
|-- requirements.txt          # Python dependencies
|-- README.md                 # Project documentation
|
|-- models/
|   |-- rf_model.pkl          # Trained Random Forest model
|   |-- preprocessor.pkl      # Fitted ColumnTransformer (StandardScaler + OneHotEncoder)
|   +-- logistic_regression.pkl
|
|-- data/
|   +-- Retail_Customer_Purchase_Prediction.csv
|
+-- notebooks/
    +-- Retail_Customer_Purchase_Prediction.ipynb
```

---

## Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/retail-purchase-prediction.git
cd retail-purchase-prediction
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Retrain the Model

To reproduce the trained models from scratch:

```bash
python train.py
```

This will:
- Load and clean the dataset from `data/`
- Engineer all features exactly as in the notebook
- Train Logistic Regression and Random Forest
- Print full evaluation metrics and classification report
- Save `rf_model.pkl`, `preprocessor.pkl`, and `logistic_regression.pkl` to `models/`

---

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click **Create app**
5. Set **Main file path** to `app.py`
6. Click **Deploy**

---

## Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11 | Language |
| scikit-learn | 1.6.1 | ML models and preprocessing |
| pandas | 3.x | Data manipulation |
| numpy | 2.x | Numerical computing |
| joblib | 1.x | Model serialisation |
| Streamlit | 1.64 | Web app deployment |

---

## Sample Prediction Output

```
Purchase Prediction:   PURCHASE
Purchase Probability:  77.00%
Purchase Likelihood:   HIGH
```

---

## Author

**Aryan**
Hackathon Project - PS-02: Retail Customer Purchase Prediction

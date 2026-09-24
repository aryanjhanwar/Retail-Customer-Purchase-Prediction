import streamlit as st
import pandas as pd
import numpy as np
import joblib

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Retail Customer Purchase Prediction",
    page_icon="cart",
    layout="wide",
)

# LOAD MODELS
@st.cache_resource
def load_model():
    model = joblib.load("models/logistic_regression.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model()

# HEADER
st.title("Retail Customer Purchase Prediction")
st.markdown(
    "This system uses a trained **Logistic Regression** model to predict whether an online "
    "retail customer is likely to make a purchase based on their website interaction, "
    "browsing behaviour, and purchase history.\n\n"
    "Fill in the customer details below and click **Predict Purchase**."
)
st.divider()

# SECTION 1: Customer Behaviour
st.subheader("1. Customer Behaviour")
col1, col2, col3 = st.columns(3)
with col1:
    pages_visited = st.number_input("Pages Visited", min_value=1, max_value=16, value=6, step=1)
    products_viewed = st.number_input("Products Viewed", min_value=0, max_value=18, value=6, step=1)
with col2:
    searches_performed = st.number_input("Searches Performed", min_value=0, max_value=11, value=1, step=1)
    product_detail_views = st.number_input("Product Detail Views", min_value=0, max_value=13, value=4, step=1)
with col3:
    time_spent = st.number_input("Time Spent (seconds)", min_value=7.0, max_value=1016.0, value=221.5, step=1.0)
    wishlist_items = st.number_input("Wishlist Items", min_value=0, max_value=10, value=1, step=1)
st.divider()

# SECTION 2: Purchase History
st.subheader("2. Purchase History")
col4, col5, col6 = st.columns(3)
with col4:
    previous_purchases = st.number_input("Previous Purchases", min_value=0, max_value=12, value=1, step=1)
    previous_total_spend = st.number_input("Previous Total Spend", min_value=0.0, max_value=2000.0, value=47.59, step=0.01)
with col5:
    average_order_value = st.number_input("Average Order Value", min_value=0.0, max_value=411.0, value=32.38, step=0.01)
    days_since_last_visit = st.number_input("Days Since Last Visit", min_value=0, max_value=191, value=24, step=1)
with col6:
    unknown_last_purchase = st.checkbox("Days Since Last Purchase unknown / first-time buyer", value=False)
    if unknown_last_purchase:
        days_since_last_purchase = 33.0
        previous_purchase_missing = 1
        st.info("Days Since Last Purchase will be filled with the dataset median (33 days).")
    else:
        days_since_last_purchase = st.number_input("Days Since Last Purchase", min_value=1.0, max_value=290.0, value=33.0, step=1.0)
        previous_purchase_missing = 0
st.divider()

# SECTION 3: Website Interaction
st.subheader("3. Website Interaction")
col7, col8, col9 = st.columns(3)
with col7:
    cart_items = st.number_input("Cart Items", min_value=0, max_value=9, value=0, step=1)
    cart_activity = st.selectbox("Cart Activity", options=["No", "Yes"])
with col8:
    discount_viewed = st.selectbox("Discount Viewed", options=["No", "Yes"])
    discount_used = st.selectbox("Discount Used", options=["No", "Yes"])
with col9:
    session_count = st.number_input("Session Count", min_value=1, max_value=11, value=3, step=1)
    previous_visits = st.number_input("Previous Visits", min_value=0, max_value=10, value=1, step=1)
st.divider()

# SECTION 4: Device / Traffic Information
st.subheader("4. Device / Traffic Information")
col10, col11 = st.columns(2)
with col10:
    device_type = st.selectbox("Device Type", options=["Desktop", "Mobile", "Tablet"])
with col11:
    traffic_source = st.selectbox("Traffic Source", options=["Direct", "Email", "Organic Search", "Paid Search", "Referral", "Social Media"])
st.divider()

# PREDICTION
if st.button("Predict Purchase", type="primary", use_container_width=True):
    # Feature Engineering (mirrors notebook exactly)
    engagement_score = int(pages_visited) + int(products_viewed) + int(product_detail_views) + int(searches_performed)
    has_cart = 1 if int(cart_items) > 0 else 0

    # Build DataFrame with EXACT feature names and order
    input_data = pd.DataFrame([{
        "Pages_Visited":            int(pages_visited),
        "Time_Spent":               float(time_spent),
        "Previous_Visits":          int(previous_visits),
        "Products_Viewed":          int(products_viewed),
        "Cart_Activity":            cart_activity,
        "Cart_Items":               int(cart_items),
        "Wishlist_Items":           int(wishlist_items),
        "Searches_Performed":       int(searches_performed),
        "Product_Detail_Views":     int(product_detail_views),
        "Discount_Viewed":          discount_viewed,
        "Discount_Used":            discount_used,
        "Previous_Purchases":       int(previous_purchases),
        "Previous_Total_Spend":     float(previous_total_spend),
        "Average_Order_Value":      float(average_order_value),
        "Days_Since_Last_Visit":    int(days_since_last_visit),
        "Days_Since_Last_Purchase": float(days_since_last_purchase),
        "Device_Type":              device_type,
        "Traffic_Source":           traffic_source,
        "Session_Count":            int(session_count),
        "Previous_Purchase_Missing": int(previous_purchase_missing),
        "Engagement_Score":         int(engagement_score),
        "Has_Cart":                 int(has_cart),
    }])

    # Transform and Predict
    input_transformed = preprocessor.transform(input_data)
    prediction        = model.predict(input_transformed)[0]
    probability       = model.predict_proba(input_transformed)[0][1]

    # Likelihood
    if probability >= 0.70:
        likelihood = "HIGH"
    elif probability >= 0.40:
        likelihood = "MEDIUM"
    else:
        likelihood = "LOW"

    # Results
    st.divider()
    st.subheader("Prediction Results")
    r1, r2, r3 = st.columns(3)
    with r1:
        if prediction == 1:
            st.success("Purchase Prediction\n\nPURCHASE")
        else:
            st.error("Purchase Prediction\n\nNO PURCHASE")
    with r2:
        st.metric(label="Purchase Probability", value=f"{probability * 100:.2f}%")
    with r3:
        if likelihood == "HIGH":
            st.success(f"Purchase Likelihood\n\n{likelihood}")
        elif likelihood == "MEDIUM":
            st.warning(f"Purchase Likelihood\n\n{likelihood}")
        else:
            st.error(f"Purchase Likelihood\n\n{likelihood}")

    st.markdown("**Purchase Probability Gauge**")
    st.progress(float(probability))
    st.caption(
        f"Probability: {probability * 100:.2f}% | "
        "Thresholds: High >= 70% | Medium 40-69% | Low < 40%"
    )

    with st.expander("Computed Feature Details"):
        st.info(
            f"Engagement Score = Pages Visited + Products Viewed + Product Detail Views + Searches Performed = {engagement_score}\n\n"
            f"Has Cart = Cart Items > 0 = {has_cart}\n\n"
            f"Previous Purchase Missing = {previous_purchase_missing}"
        )

st.divider()
st.caption("PS-02 - Retail Customer Purchase Prediction | Model: Logistic Regression")

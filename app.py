import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="SmartCart Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

# Title
st.title("🛒 SmartCart - E-commerce Customer Segmentation System")
st.write("Predict customer segments using the trained KMeans clustering model.")

# Load Models
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
kmeans = joblib.load("kmeans_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Input Section
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    education = st.selectbox(
        "Education",
        ["Graduate", "Postgraduate", "Undergraduate"]
    )

    income = st.number_input("Income", min_value=0.0)

    recency = st.number_input("Recency", min_value=0)

    num_deals = st.number_input(
        "NumDealsPurchases",
        min_value=0
    )

    num_web = st.number_input(
        "NumWebPurchases",
        min_value=0
    )

    num_catalog = st.number_input(
        "NumCatalogPurchases",
        min_value=0
    )

    num_store = st.number_input(
        "NumStorePurchases",
        min_value=0
    )

with col2:
    num_web_visits = st.number_input(
        "NumWebVisitsMonth",
        min_value=0
    )

    complain = st.selectbox(
        "Complain",
        [0, 1]
    )

    response = st.selectbox(
        "Response",
        [0, 1]
    )

    age = st.number_input(
        "Age",
        min_value=0
    )

    tenure = st.number_input(
        "Customer_Tenure_Days",
        min_value=0
    )

    total_spending = st.number_input(
        "Total_Spending",
        min_value=0.0
    )

    total_children = st.number_input(
        "Total_Children",
        min_value=0
    )

    living_with = st.selectbox(
        "Living_With",
        ["Alone", "Partner"]
    )

# Prediction Button
if st.button("Predict Customer Segment"):

    # Create Input DataFrame
    input_df = pd.DataFrame({
        "Education": [education],
        "Income": [income],
        "Recency": [recency],
        "NumDealsPurchases": [num_deals],
        "NumWebPurchases": [num_web],
        "NumCatalogPurchases": [num_catalog],
        "NumStorePurchases": [num_store],
        "NumWebVisitsMonth": [num_web_visits],
        "Complain": [complain],
        "Response": [response],
        "Age": [age],
        "Customer_Tenure_Days": [tenure],
        "Total_Spending": [total_spending],
        "Total_Children": [total_children],
        "Living_With": [living_with]
    })

    # Encode Categorical Features
    cat_cols = ["Education", "Living_With"]

    encoded = encoder.transform(input_df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded.toarray(),
        columns=encoder.get_feature_names_out(cat_cols)
    )

    # Merge Encoded + Numeric Data
    final_df = pd.concat(
        [
            input_df.drop(columns=cat_cols).reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )

    # Match Training Columns
    final_df = final_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scaling
    scaled_data = scaler.transform(final_df)

    # PCA
    pca_data = pca.transform(scaled_data)

    # Prediction
    cluster = int(kmeans.predict(pca_data)[0])

    st.success(f"Predicted Customer Segment: Cluster {cluster}")

    # Cluster Interpretation
    if cluster == 0:
        st.info("Cluster 0 : Low to Moderate Spending Customers")

    elif cluster == 1:
        st.info("Cluster 1 : High Value Customers")

    elif cluster == 2:
        st.info("Cluster 2 : Regular Customers")

    elif cluster == 3:
        st.info("Cluster 3 : Premium Customers")

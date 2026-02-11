import os
import streamlit as st
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
LOCAL_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Telco-Customer-Churn.csv",
)


@st.cache_data(show_spinner="Loading dataset...")
def load_data(uploaded_file=None):
    """Load and clean the Telco Customer Churn dataset."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists(LOCAL_CSV):
        df = pd.read_csv(LOCAL_CSV)
    else:
        try:
            df = pd.read_csv(DATA_URL)
        except Exception:
            return None

    # Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges has blank strings — convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert SeniorCitizen to Yes/No for consistency
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def get_feature_groups():
    """Return feature groups for selection."""
    return {
        "Demographics": ["gender", "SeniorCitizen", "Partner", "Dependents"],
        "Account Info": ["tenure", "Contract", "PaperlessBilling", "PaymentMethod"],
        "Services": [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ],
        "Billing": ["MonthlyCharges", "TotalCharges"],
    }


def get_column_descriptions():
    """Return column descriptions grouped by category."""
    return {
        "Demographics": {
            "gender": "Customer gender (Male/Female)",
            "SeniorCitizen": "Whether the customer is a senior citizen (Yes/No)",
            "Partner": "Whether the customer has a partner (Yes/No)",
            "Dependents": "Whether the customer has dependents (Yes/No)",
        },
        "Account": {
            "tenure": "Number of months the customer has stayed with the company",
            "Contract": "Contract term (Month-to-month, One year, Two year)",
            "PaperlessBilling": "Whether the customer has paperless billing (Yes/No)",
            "PaymentMethod": "Payment method (Electronic check, Mailed check, etc.)",
        },
        "Services": {
            "PhoneService": "Whether the customer has phone service (Yes/No)",
            "MultipleLines": "Whether the customer has multiple lines",
            "InternetService": "Customer's internet service provider (DSL, Fiber optic, No)",
            "OnlineSecurity": "Whether the customer has online security",
            "OnlineBackup": "Whether the customer has online backup",
            "DeviceProtection": "Whether the customer has device protection",
            "TechSupport": "Whether the customer has tech support",
            "StreamingTV": "Whether the customer has streaming TV",
            "StreamingMovies": "Whether the customer has streaming movies",
        },
        "Billing": {
            "MonthlyCharges": "The amount charged to the customer monthly",
            "TotalCharges": "The total amount charged to the customer",
        },
        "Target": {
            "Churn": "Whether the customer churned (Yes/No)",
        },
    }

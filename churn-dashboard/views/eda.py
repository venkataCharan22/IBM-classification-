import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.preprocessing import encode_features
from utils.plotting import (
    churn_distribution_charts, churn_by_contract, tenure_histogram,
    monthly_charges_box, service_impact_chart, internet_service_churn,
    correlation_heatmap, top_churn_predictors,
)


def render():
    st.markdown(
        "<h1 style='color:#0EA5E9; font-weight:800;'>📊 Exploratory Data Analysis</h1>",
        unsafe_allow_html=True,
    )

    df = load_data()
    if df is None:
        st.error("Dataset not loaded. Please visit the Overview page first.")
        return

    # ---- 1. Churn Distribution ----
    st.subheader("1. Churn Distribution")
    fig_pie, fig_bar = churn_distribution_charts(df)
    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_pie, use_container_width=True)
    c2.plotly_chart(fig_bar, use_container_width=True)
    st.info(
        "The dataset is **imbalanced** — roughly 73% of customers did not churn, "
        "while 27% did. This imbalance must be handled during preprocessing."
    )

    st.markdown("---")

    # ---- 2. Churn by Contract Type ----
    st.subheader("2. Churn by Contract Type")
    st.plotly_chart(churn_by_contract(df), use_container_width=True)
    st.info(
        "**Month-to-month** contracts have the highest churn rate (~42%). "
        "Customers on **two-year** contracts rarely churn (~3%), indicating that "
        "long-term contracts are a strong retention mechanism."
    )

    st.markdown("---")

    # ---- 3. Tenure Distribution ----
    st.subheader("3. Tenure Distribution")
    st.plotly_chart(tenure_histogram(df), use_container_width=True)
    st.info(
        "Most churned customers leave within the **first 12 months**. "
        "The first year is the critical period for customer retention."
    )

    st.markdown("---")

    # ---- 4. Monthly Charges vs Churn ----
    st.subheader("4. Monthly Charges vs Churn")
    st.plotly_chart(monthly_charges_box(df), use_container_width=True)
    st.info(
        "Churned customers tend to have **higher monthly charges**. "
        "The median monthly charge for churned customers is significantly higher "
        "than for retained customers."
    )

    st.markdown("---")

    # ---- 5. Service Impact ----
    st.subheader("5. Service Impact on Churn")
    st.plotly_chart(service_impact_chart(df), use_container_width=True)
    st.info(
        "Customers **without** Online Security, Tech Support, Online Backup, or "
        "Device Protection have substantially higher churn rates. These services "
        "act as retention anchors."
    )

    st.markdown("---")

    # ---- 6. Internet Service Type ----
    st.subheader("6. Churn by Internet Service Type")
    st.plotly_chart(internet_service_churn(df), use_container_width=True)
    st.info(
        "**Fiber optic** customers churn at nearly double the rate of DSL customers. "
        "This may be due to higher costs or competition in fiber markets."
    )

    st.markdown("---")

    # ---- 7. Correlation Heatmap ----
    st.subheader("7. Correlation Heatmap")
    df_encoded, _ = encode_features(df)
    st.plotly_chart(correlation_heatmap(df_encoded), use_container_width=True)
    st.info(
        "The heatmap reveals relationships between features. "
        "Tenure and TotalCharges are highly correlated, which is expected."
    )

    st.markdown("---")

    # ---- 8. Top Churn Predictors ----
    st.subheader("8. Top Churn Predictors")
    st.plotly_chart(top_churn_predictors(df_encoded), use_container_width=True)
    st.info(
        "Contract type, tenure, OnlineSecurity, and TechSupport show the strongest "
        "correlation with churn. These are the features our models will rely on most."
    )

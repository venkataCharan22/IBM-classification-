import streamlit as st
from utils.data_loader import load_data, get_column_descriptions


def render():
    # ---- Hero Section ----
    st.markdown(
        """
        <div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
            <h1 style="color:#0EA5E9; font-size:2.6rem; margin-bottom:0.3rem;
                        letter-spacing:-0.02em; font-weight:800;">
                Customer Churn Prediction
            </h1>
            <p style="color:#94A3B8; font-size:1.1rem; max-width:600px; margin:0 auto;">
                Predicting telecom customer churn using classification algorithms
                &mdash; IBM Telco Dataset
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Load Data ----
    df = load_data()
    if df is None:
        st.warning("Could not load data from URL. Please upload the CSV manually.")
        uploaded = st.file_uploader("Upload Telco-Customer-Churn.csv", type="csv")
        if uploaded:
            df = load_data(uploaded_file=uploaded)
        else:
            return

    st.markdown("")  # spacer

    # ---- Key Stats ----
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Features", f"{df.shape[1] - 1}")
    col3.metric("Churn Rate", f"{churn_rate:.1f}%")
    col4.metric("Retained Rate", f"{100 - churn_rate:.1f}%")

    st.markdown("---")

    # ---- Dataset Preview ----
    st.markdown("#### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, height=380)

    st.markdown("---")

    # ---- Column Descriptions ----
    st.markdown("#### Feature Descriptions")
    descriptions = get_column_descriptions()

    cols = st.columns(2)
    categories = list(descriptions.items())
    for idx, (category, features) in enumerate(categories):
        with cols[idx % 2]:
            with st.expander(f"**{category}**", expanded=(idx < 2)):
                for feat, desc in features.items():
                    st.markdown(f"- **`{feat}`**: {desc}")

    st.markdown("---")

    # ---- Basic Statistics ----
    st.markdown("#### Statistical Summary")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

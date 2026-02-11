import streamlit as st

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Global Custom CSS ----
st.markdown("""
<style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(14, 165, 233, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0EA5E9 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Section dividers */
    hr {
        border-color: rgba(14, 165, 233, 0.15) !important;
        margin: 1.5rem 0 !important;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        border: 1px solid rgba(14, 165, 233, 0.15) !important;
        border-radius: 10px !important;
        background-color: rgba(30, 41, 59, 0.4) !important;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0EA5E9 0%, #06B6D4 100%) !important;
        border: none !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #06B6D4 0%, #0EA5E9 100%) !important;
        box-shadow: 0 4px 20px rgba(14, 165, 233, 0.4) !important;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
        border-right: 1px solid rgba(14, 165, 233, 0.1);
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 6px 12px !important;
        border-radius: 6px;
        transition: background 0.2s;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(14, 165, 233, 0.1);
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important;
        letter-spacing: 0.02em;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #0EA5E9 !important;
    }

    /* Info/Warning/Success boxes */
    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    /* Remove extra top padding */
    .block-container {
        padding-top: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- Sidebar Navigation ----
st.sidebar.markdown(
    "<h2 style='color:#0EA5E9; margin-bottom:0;'>📉 Churn Predictor</h2>",
    unsafe_allow_html=True,
)
st.sidebar.caption("IBM Telco Customer Churn Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📊 EDA",
        "⚙️ Preprocessing Lab",
        "🤖 Model Arena",
        "📈 Overfitting Lab",
        "🔄 Cross-Validation",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center; color:#64748B; font-size:0.8rem;'>"
    "Built for IBM Internship<br>Presentation"
    "</div>",
    unsafe_allow_html=True,
)

# ---- Page Router ----
if page == "🏠 Overview":
    from views import overview
    overview.render()
elif page == "📊 EDA":
    from views import eda
    eda.render()
elif page == "⚙️ Preprocessing Lab":
    from views import preprocessing_lab
    preprocessing_lab.render()
elif page == "🤖 Model Arena":
    from views import model_arena
    model_arena.render()
elif page == "📈 Overfitting Lab":
    from views import overfitting_lab
    overfitting_lab.render()
elif page == "🔄 Cross-Validation":
    from views import cross_validation
    cross_validation.render()

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data, get_feature_groups
from utils.preprocessing import prepare_data
from utils.models import train_all_models, MODEL_CONFIGS
from utils.plotting import (
    confusion_matrix_chart, model_comparison_bar, f1_ranking_chart, COLORS,
)


@st.cache_resource(show_spinner=False)
def cached_train_all(X_train_bytes, y_train_bytes, X_test_bytes, y_test_bytes,
                     feature_names, random_state):
    """Cache trained models using hashable byte representations."""
    import pickle
    X_train = pickle.loads(X_train_bytes)
    y_train = pickle.loads(y_train_bytes)
    X_test = pickle.loads(X_test_bytes)
    y_test = pickle.loads(y_test_bytes)
    return train_all_models(X_train, y_train, X_test, y_test, random_state)


def render():
    st.markdown(
        "<h1 style='color:#0EA5E9; font-weight:800;'>🤖 Model Arena</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("Train, tune, and compare **5 classification algorithms** head-to-head.")

    df = load_data()
    if df is None:
        st.error("Dataset not loaded. Please visit the Overview page first.")
        return

    st.markdown("---")

    # ---- Preprocessing (use sensible defaults) ----
    with st.expander("⚙️ Preprocessing Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, key="arena_test")
        with c2:
            use_smote = st.toggle("SMOTE", True, key="arena_smote")
            use_scaling = st.toggle("Scaling", True, key="arena_scaling")
        with c3:
            random_state = st.number_input("Random State", value=42, key="arena_rs")

    all_groups = list(get_feature_groups().keys())

    if st.button("🏟️ Train All Models", type="primary", use_container_width=True):
        result = prepare_data(
            df, all_groups, test_size=test_size,
            use_smote=use_smote, use_scaling=use_scaling,
            random_state=random_state,
        )
        if result is None:
            st.error("Preprocessing failed.")
            return

        import pickle
        with st.spinner("Training 5 models with hyperparameter tuning... This may take a moment."):
            results = cached_train_all(
                pickle.dumps(result["X_train"]),
                pickle.dumps(result["y_train"]),
                pickle.dumps(result["X_test"]),
                pickle.dumps(result["y_test"]),
                tuple(result["feature_names"]),
                random_state,
            )
        st.session_state["arena_results"] = results

    # ---- Display Results ----
    if "arena_results" not in st.session_state:
        st.info("Click **Train All Models** to begin.")
        return

    results = st.session_state["arena_results"]

    st.markdown("---")

    # ---- Individual Model Cards ----
    st.subheader("Individual Model Results")

    for name, res in results.items():
        with st.expander(f"**{name}** — _{res['tagline']}_", expanded=False):
            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown("**Best Hyperparameters:**")
                for param, val in res["best_params"].items():
                    st.markdown(f"- `{param}`: **{val}**")

                st.markdown("**Metrics:**")
                m1, m2 = st.columns(2)
                m1.metric("Accuracy", f"{res['metrics']['accuracy']:.4f}")
                m2.metric("Precision", f"{res['metrics']['precision']:.4f}")
                m1.metric("Recall", f"{res['metrics']['recall']:.4f}")
                m2.metric("F1 Score", f"{res['metrics']['f1']:.4f}")

            with c2:
                st.plotly_chart(
                    confusion_matrix_chart(res["confusion_matrix"], f"{name} Confusion Matrix"),
                    use_container_width=True,
                )

            # Classification report
            report_df = pd.DataFrame(res["classification_report"]).T
            report_df = report_df.drop(index=["accuracy"], errors="ignore")
            st.markdown("**Classification Report:**")
            st.dataframe(
                report_df.style.format("{:.4f}").background_gradient(
                    cmap="Blues", subset=["precision", "recall", "f1-score"]
                ),
                use_container_width=True,
            )

            # Insight
            if name == "KNN":
                st.info(
                    "KNN classifies based on the majority vote of nearest neighbors. "
                    "Its performance depends heavily on feature scaling and the choice of K."
                )
            elif name == "Decision Tree":
                st.info(
                    "Decision Trees are interpretable but prone to overfitting. "
                    "Pruning via max_depth helps control complexity."
                )
            elif name == "Random Forest":
                st.info(
                    "Random Forest aggregates many decision trees to reduce variance. "
                    "It's typically the most robust classifier for tabular data."
                )
            elif name == "Naive Bayes":
                st.info(
                    "Naive Bayes assumes feature independence. It's fast and works well "
                    "as a baseline, but the independence assumption is rarely true."
                )
            elif name == "SVM":
                st.info(
                    "SVM finds the optimal hyperplane to separate classes. "
                    "It works well in high-dimensional spaces but is slower to train."
                )

    st.markdown("---")

    # ---- Comparison Section ----
    st.subheader("Model Comparison")

    # Metrics table
    comparison_data = {
        name: res["metrics"] for name, res in results.items()
    }
    comp_df = pd.DataFrame(comparison_data).T
    comp_df.columns = ["Accuracy", "Precision", "Recall", "F1 Score"]
    comp_df = comp_df.sort_values("F1 Score", ascending=False)

    st.dataframe(
        comp_df.style.format("{:.4f}").background_gradient(
            cmap="Blues", axis=0
        ),
        use_container_width=True,
    )

    # Charts
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(model_comparison_bar(results), use_container_width=True)
    with c2:
        st.plotly_chart(f1_ranking_chart(results), use_container_width=True)

    # Winner
    winner = comp_df.index[0]
    winner_f1 = comp_df.iloc[0]["F1 Score"]
    st.success(
        f"🏆 **Winner: {winner}** with an F1 Score of **{winner_f1:.4f}**. "
        f"This model achieves the best balance of precision and recall for churn prediction."
    )

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data, get_feature_groups
from utils.preprocessing import prepare_data, encode_features
from utils.plotting import class_distribution_bar, feature_distribution_chart, COLORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def render():
    st.markdown(
        "<h1 style='color:#0EA5E9; font-weight:800;'>⚙️ Preprocessing Lab</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Experiment with different preprocessing choices and see how they "
        "affect model performance **in real-time**."
    )

    df = load_data()
    if df is None:
        st.error("Dataset not loaded. Please visit the Overview page first.")
        return

    # ---- Controls ----
    st.markdown("---")
    st.markdown("#### Configuration")
    c1, c2, c3 = st.columns(3)

    with c1:
        test_size = st.slider(
            "Test Size (%)", min_value=10, max_value=50, value=20, step=5
        ) / 100
        random_state = st.number_input("Random State", value=42, min_value=0, max_value=999)

    with c2:
        use_smote = st.toggle("Enable SMOTE Oversampling", value=False)
        use_scaling = st.toggle("Enable Feature Scaling (StandardScaler)", value=False)

    with c3:
        all_groups = get_feature_groups()
        feature_groups = st.multiselect(
            "Feature Groups to Include",
            options=list(all_groups.keys()),
            default=list(all_groups.keys()),
        )

    if not feature_groups:
        st.warning("Please select at least one feature group.")
        return

    if use_smote:
        st.warning(
            "**SMOTE is applied ONLY to training data**, never to test data. "
            "This prevents data leakage and gives an honest evaluation."
        )

    st.markdown("")

    # ---- Run Button ----
    if st.button("Run Preprocessing & Quick Model", type="primary", use_container_width=True):
        with st.spinner("Running preprocessing pipeline..."):
            result = prepare_data(
                df, feature_groups, test_size=test_size,
                use_smote=use_smote, use_scaling=use_scaling,
                random_state=random_state,
            )

        if result is None:
            st.error("No features selected. Check your feature groups.")
            return

        st.session_state["preprocess_result"] = result
        st.session_state["preprocess_config"] = {
            "test_size": test_size,
            "use_smote": use_smote,
            "use_scaling": use_scaling,
            "feature_groups": feature_groups,
            "random_state": random_state,
        }

        # Train quick Random Forest
        with st.spinner("Training Quick Random Forest..."):
            rf = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1
            )
            rf.fit(result["X_train"], result["y_train"])
            y_pred = rf.predict(result["X_test"])
            metrics = {
                "Accuracy": accuracy_score(result["y_test"], y_pred),
                "Precision": precision_score(result["y_test"], y_pred),
                "Recall": recall_score(result["y_test"], y_pred),
                "F1 Score": f1_score(result["y_test"], y_pred),
            }
            st.session_state["preprocess_metrics"] = metrics

    # ---- Display Results ----
    if "preprocess_result" in st.session_state:
        result = st.session_state["preprocess_result"]
        config = st.session_state["preprocess_config"]

        st.markdown("---")

        # Split sizes
        st.markdown("#### Train / Test Split")
        s1, s2, s3 = st.columns(3)
        s1.metric("Training Samples", f"{len(result['X_train']):,}")
        s2.metric("Test Samples", f"{len(result['X_test']):,}")
        s3.metric("Features Used", f"{len(result['feature_names'])}")

        st.markdown("")

        # Class distribution
        st.markdown("#### Class Distribution")
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(
                class_distribution_bar(
                    result["class_dist_before"], "Before SMOTE (Full Dataset)"
                ),
                use_container_width=True,
            )
        with d2:
            title = "After SMOTE (Training Data)" if config["use_smote"] else "Training Data (No SMOTE)"
            st.plotly_chart(
                class_distribution_bar(result["class_dist_after"], title),
                use_container_width=True,
            )

        # Scaled vs Unscaled
        if config["use_scaling"]:
            st.markdown("#### Feature Scaling Effect")
            if "tenure" in result["feature_names"]:
                feat = "tenure"
            elif "MonthlyCharges" in result["feature_names"]:
                feat = "MonthlyCharges"
            else:
                feat = result["feature_names"][0]

            df_enc, _ = encode_features(df)
            unscaled = df_enc[feat].values
            scaled = result["X_train"][feat].values

            st.plotly_chart(
                feature_distribution_chart(unscaled, scaled, feat),
                use_container_width=True,
            )
            st.info(
                "After scaling, features have **mean ~ 0** and **std ~ 1**. "
                "This is critical for distance-based algorithms like KNN and SVM."
            )

        # Metrics
        if "preprocess_metrics" in st.session_state:
            st.markdown("#### Quick Random Forest Performance")
            metrics = st.session_state["preprocess_metrics"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            m2.metric("Precision", f"{metrics['Precision']:.4f}")
            m3.metric("Recall", f"{metrics['Recall']:.4f}")
            m4.metric("F1 Score", f"{metrics['F1 Score']:.4f}")

            st.markdown("")

            # Insights
            st.markdown("#### Insights")
            if not config["use_smote"]:
                st.warning(
                    "**Without SMOTE**, the model tends to be biased toward predicting "
                    "'Not Churned' since that class is ~73% of the data. Notice the recall "
                    "may be low — meaning many actual churners are missed."
                )
            else:
                st.success(
                    "**With SMOTE**, the training classes are balanced, so the model "
                    "learns to identify churners better. Recall should improve, "
                    "though precision may decrease slightly."
                )

            if not config["use_scaling"]:
                st.info(
                    "**Without scaling**, tree-based models (like this Random Forest) "
                    "are unaffected, but KNN and SVM would perform worse because they "
                    "rely on distance calculations."
                )
            else:
                st.info(
                    "**With scaling**, all features contribute equally to distance "
                    "calculations. This is essential for KNN and SVM."
                )

            if config["test_size"] > 0.35:
                st.warning(
                    "**Large test size** means less data for training. "
                    "The model may underperform because it has fewer examples to learn from."
                )

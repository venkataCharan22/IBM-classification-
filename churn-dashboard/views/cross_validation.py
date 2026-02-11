import streamlit as st
import numpy as np
import pandas as pd
from utils.data_loader import load_data, get_feature_groups
from utils.preprocessing import encode_features
from utils.models import run_cross_validation, MODEL_CONFIGS
from utils.plotting import cv_fold_scores_chart, COLORS
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go


def render():
    st.markdown(
        "<h1 style='color:#0EA5E9; font-weight:800;'>🔄 Cross-Validation Explorer</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Understand why **cross-validation** gives more reliable estimates "
        "than a single train/test split."
    )

    df = load_data()
    if df is None:
        st.error("Dataset not loaded. Please visit the Overview page first.")
        return

    # Encode data
    df_enc, _ = encode_features(df)
    X = df_enc.drop(columns=["Churn"])
    y = df_enc["Churn"]

    tab1, tab2 = st.tabs(["📊 K-Fold CV Demo", "📏 Stratified vs Regular"])

    # ---- Tab 1: K-Fold CV Demo ----
    with tab1:
        st.subheader("K-Fold Cross-Validation")

        c1, c2, c3 = st.columns(3)
        with c1:
            n_folds = st.slider("Number of Folds (K)", 2, 15, 5)
        with c2:
            model_name = st.selectbox("Model", list(MODEL_CONFIGS.keys()))
        with c3:
            scoring = st.selectbox("Scoring Metric", ["f1", "accuracy", "recall", "precision"])

        if st.button("🔄 Run Cross-Validation", type="primary", use_container_width=True):
            with st.spinner(f"Running {n_folds}-fold CV with {model_name}..."):
                scores = run_cross_validation(
                    model_name, X, y, n_folds=n_folds, scoring=scoring, random_state=42
                )
                st.session_state["cv_scores"] = scores
                st.session_state["cv_config"] = {
                    "n_folds": n_folds,
                    "model": model_name,
                    "scoring": scoring,
                }

        if "cv_scores" in st.session_state:
            scores = st.session_state["cv_scores"]
            config = st.session_state["cv_config"]

            # Big metric
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Mean {config['scoring'].capitalize()}", f"{scores.mean():.4f}")
            m2.metric("Std Deviation", f"± {scores.std():.4f}")
            m3.metric("Score Range", f"{scores.min():.4f} — {scores.max():.4f}")

            # Fold scores chart
            st.plotly_chart(
                cv_fold_scores_chart(scores, config["n_folds"]),
                use_container_width=True,
            )

            # CV vs Hold-out comparison
            st.markdown("#### Hold-out vs Cross-Validation")
            st.markdown(
                "Let's compare: a single 80/20 split vs the CV result."
            )

            # Run single split for comparison
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.svm import SVC
            from sklearn.metrics import f1_score, recall_score, precision_score

            model_map = {
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1),
                "Naive Bayes": GaussianNB(),
                "SVM": SVC(random_state=42),
            }
            scorer_map = {
                "accuracy": accuracy_score,
                "f1": f1_score,
                "recall": recall_score,
                "precision": precision_score,
            }

            mdl = model_map[config["model"]]
            mdl.fit(X_train, y_train)
            y_pred = mdl.predict(X_test)
            holdout_score = scorer_map[config["scoring"]](y_test, y_pred)

            h1, h2 = st.columns(2)
            h1.metric("Hold-out Score (single split)", f"{holdout_score:.4f}")
            h2.metric(f"CV Mean Score ({config['n_folds']} folds)", f"{scores.mean():.4f}")

            st.info(
                "**Cross-validation is more reliable** because it tests the model on "
                f"**{config['n_folds']} different splits** instead of just one. "
                "A single hold-out split can be lucky or unlucky depending on which "
                "samples end up in the test set. CV gives a mean ± std, quantifying "
                "the model's stability."
            )

            # Visual explanation of K-fold
            with st.expander("💡 How K-Fold Cross-Validation Works"):
                st.markdown(f"""
**{config['n_folds']}-Fold Cross-Validation Process:**

1. The dataset is split into **{config['n_folds']} equal parts** (folds)
2. For each iteration:
   - **1 fold** is held out as the test set
   - The remaining **{config['n_folds'] - 1} folds** are used for training
3. The model is trained and evaluated **{config['n_folds']} times**
4. The final score is the **mean** of all {config['n_folds']} scores

**Why it's better than a single split:**
- Uses **all data** for both training and testing
- Every sample gets to be in the test set exactly once
- Reduces variance in the performance estimate
- Detects if the model is sensitive to the choice of train/test split
                """)

                # Visual diagram
                fig = go.Figure()
                for fold in range(min(config["n_folds"], 8)):
                    colors = [COLORS["cyan"]] * config["n_folds"]
                    colors[fold] = COLORS["red"]
                    for j in range(config["n_folds"]):
                        fig.add_shape(
                            type="rect",
                            x0=j * 1.1, x1=j * 1.1 + 1,
                            y0=fold * 1.3, y1=fold * 1.3 + 1,
                            fillcolor=colors[j],
                            line=dict(color=COLORS["light"], width=1),
                            opacity=0.7,
                        )
                        label = "Test" if j == fold else "Train"
                        fig.add_annotation(
                            x=j * 1.1 + 0.5, y=fold * 1.3 + 0.5,
                            text=label, showarrow=False,
                            font=dict(color="white", size=10),
                        )
                    fig.add_annotation(
                        x=config["n_folds"] * 1.1 + 0.5,
                        y=fold * 1.3 + 0.5,
                        text=f"Score: {scores[fold]:.3f}" if fold < len(scores) else "",
                        showarrow=False,
                        font=dict(color=COLORS["light"], size=11),
                    )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    height=max(200, min(config["n_folds"], 8) * 60 + 40),
                    margin=dict(l=10, r=100, t=10, b=10),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False, autorange="reversed"),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Configure settings and click **Run Cross-Validation** to begin.")

    # ---- Tab 2: Stratified vs Regular ----
    with tab2:
        st.subheader("Stratified vs Regular K-Fold Split")
        st.markdown(
            "With **imbalanced data** (27% churn), random splits might create folds "
            "where one class is underrepresented. **Stratified** splitting ensures each "
            "fold maintains the original class ratio."
        )

        n_demo_folds = st.slider("Folds for demo", 3, 10, 5, key="strat_folds")

        if st.button("📊 Compare Splits", type="primary"):
            from sklearn.model_selection import KFold

            # Regular KFold
            regular_kf = KFold(n_splits=n_demo_folds, shuffle=True, random_state=42)
            regular_dists = []
            for train_idx, test_idx in regular_kf.split(X, y):
                fold_y = y.iloc[test_idx]
                churn_pct = (fold_y == 1).mean() * 100
                regular_dists.append(churn_pct)

            # Stratified KFold
            strat_kf = StratifiedKFold(n_splits=n_demo_folds, shuffle=True, random_state=42)
            strat_dists = []
            for train_idx, test_idx in strat_kf.split(X, y):
                fold_y = y.iloc[test_idx]
                churn_pct = (fold_y == 1).mean() * 100
                strat_dists.append(churn_pct)

            st.session_state["split_comparison"] = {
                "regular": regular_dists,
                "stratified": strat_dists,
                "n_folds": n_demo_folds,
            }

        if "split_comparison" in st.session_state:
            comp = st.session_state["split_comparison"]
            fold_labels = [f"Fold {i+1}" for i in range(comp["n_folds"])]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=fold_labels, y=comp["regular"],
                name="Regular KFold",
                marker_color=COLORS["orange"],
                text=[f"{v:.1f}%" for v in comp["regular"]],
                textposition="outside",
            ))
            fig.add_trace(go.Bar(
                x=fold_labels, y=comp["stratified"],
                name="Stratified KFold",
                marker_color=COLORS["green"],
                text=[f"{v:.1f}%" for v in comp["stratified"]],
                textposition="outside",
            ))
            fig.add_hline(
                y=26.5, line_dash="dash", line_color=COLORS["red"],
                annotation_text="Original churn rate (~26.5%)",
                annotation_font_color=COLORS["red"],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=COLORS["light"]),
                barmode="group",
                title="Churn % in Each Fold's Test Set",
                yaxis_title="Churn Rate (%)",
                margin=dict(l=40, r=40, t=50, b=40),
                height=450,
            )
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
            st.plotly_chart(fig, use_container_width=True)

            # Stats
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Regular KFold**")
                st.metric("Std Dev of Churn %", f"{np.std(comp['regular']):.2f}%")
            with c2:
                st.markdown("**Stratified KFold**")
                st.metric("Std Dev of Churn %", f"{np.std(comp['stratified']):.2f}%")

            st.success(
                "**Stratified K-Fold** maintains consistent class distribution across folds. "
                "This is critical for imbalanced datasets like ours — without stratification, "
                "some folds might have very few churn examples, leading to unreliable metrics."
            )

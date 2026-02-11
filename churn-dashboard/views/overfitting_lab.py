import streamlit as st
import numpy as np
from utils.data_loader import load_data, get_feature_groups
from utils.preprocessing import prepare_data
from utils.models import (
    get_overfitting_data_dt, get_overfitting_data_knn,
    get_overfitting_data_rf, get_learning_curve_data,
)
from utils.plotting import (
    overfitting_line_chart, confusion_matrix_chart,
    learning_curve_chart, COLORS,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


@st.cache_data(show_spinner="Computing overfitting curves...")
def compute_all_curves(_X_train, _y_train, _X_test, _y_test, random_state):
    """Pre-compute all overfitting data."""
    dt_depths, dt_train, dt_test = get_overfitting_data_dt(
        _X_train, _y_train, _X_test, _y_test, range(1, 31), random_state
    )
    knn_ks, knn_train, knn_test = get_overfitting_data_knn(
        _X_train, _y_train, _X_test, _y_test, range(1, 51)
    )
    rf_ns, rf_train, rf_test = get_overfitting_data_rf(
        _X_train, _y_train, _X_test, _y_test,
        list(range(10, 301, 10)), random_state
    )
    return {
        "dt": (dt_depths, dt_train, dt_test),
        "knn": (knn_ks, knn_train, knn_test),
        "rf": (rf_ns, rf_train, rf_test),
    }


def render():
    st.markdown(
        "<h1 style='color:#0EA5E9; font-weight:800;'>📈 Overfitting & Underfitting Lab</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Watch models **overfit and underfit** as you adjust their complexity. "
        "The gap between training and test accuracy tells the story."
    )

    df = load_data()
    if df is None:
        st.error("Dataset not loaded. Please visit the Overview page first.")
        return

    # Prepare data with default settings
    all_groups = list(get_feature_groups().keys())
    result = prepare_data(
        df, all_groups, test_size=0.2,
        use_smote=True, use_scaling=True, random_state=42,
    )
    if result is None:
        st.error("Preprocessing failed.")
        return

    X_train = result["X_train"]
    y_train = result["y_train"]
    X_test = result["X_test"]
    y_test = result["y_test"]

    # Pre-compute curves
    curves = compute_all_curves(X_train, y_train, X_test, y_test, 42)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🌳 Decision Tree Depth",
        "🔍 KNN K-Value",
        "🌲 Random Forest Trees",
        "📚 Learning Curves",
    ])

    # ---- Tab 1: Decision Tree Depth ----
    with tab1:
        st.subheader("Decision Tree: Effect of max_depth")

        dt_depths, dt_train, dt_test = curves["dt"]

        depth = st.slider("Select max_depth", 1, 30, 5, key="dt_depth")

        # Show line chart with all depths
        fig = overfitting_line_chart(
            dt_depths, dt_train, dt_test,
            "max_depth", "Decision Tree: Training vs Test Accuracy"
        )
        # Add a marker for current selection
        idx = depth - 1
        fig.add_scatter(
            x=[depth], y=[dt_test[idx]],
            mode="markers", marker=dict(size=15, color=COLORS["orange"], symbol="star"),
            name=f"Selected (depth={depth})",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Metrics for selected depth
        c1, c2, c3 = st.columns(3)
        c1.metric("Training Accuracy", f"{dt_train[idx]:.4f}")
        c2.metric("Test Accuracy", f"{dt_test[idx]:.4f}")
        gap = dt_train[idx] - dt_test[idx]
        c3.metric("Gap (Overfit Indicator)", f"{gap:.4f}",
                  delta=f"{gap:.4f}", delta_color="inverse")

        # Confusion matrix at selected depth
        dt_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt_model.fit(X_train, y_train)
        cm = confusion_matrix(y_test, dt_model.predict(X_test))
        st.plotly_chart(
            confusion_matrix_chart(cm, f"Confusion Matrix (depth={depth})"),
            use_container_width=True,
        )

        # Dynamic explanation
        if depth <= 3:
            st.warning(
                "🔻 **Underfitting** — The tree is too simple and misses important patterns. "
                "Both training and test accuracy are low."
            )
        elif depth <= 8:
            st.success(
                "✅ **Good Fit** — The tree captures meaningful patterns without memorizing noise. "
                "Training and test accuracy are close together."
            )
        else:
            st.error(
                "🔺 **Overfitting** — The tree is memorizing training data (high training accuracy) "
                "but failing on new data (test accuracy drops or plateaus). "
                "The growing gap between the two lines is the hallmark of overfitting."
            )

    # ---- Tab 2: KNN K-Value ----
    with tab2:
        st.subheader("KNN: Effect of K (Number of Neighbors)")

        knn_ks, knn_train, knn_test = curves["knn"]

        k_val = st.slider("Select K", 1, 50, 5, key="knn_k")

        fig = overfitting_line_chart(
            knn_ks, knn_train, knn_test,
            "K", "KNN: Training vs Test Accuracy"
        )
        k_idx = k_val - 1
        fig.add_scatter(
            x=[k_val], y=[knn_test[k_idx]],
            mode="markers", marker=dict(size=15, color=COLORS["orange"], symbol="star"),
            name=f"Selected (K={k_val})",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Training Accuracy", f"{knn_train[k_idx]:.4f}")
        c2.metric("Test Accuracy", f"{knn_test[k_idx]:.4f}")
        gap = knn_train[k_idx] - knn_test[k_idx]
        c3.metric("Gap", f"{gap:.4f}", delta=f"{gap:.4f}", delta_color="inverse")

        if k_val <= 3:
            st.error(
                "🔺 **Overfitting** — With very few neighbors (especially K=1), "
                "the model memorizes training data. Each point is its own decision boundary."
            )
        elif k_val <= 15:
            st.success(
                "✅ **Good Fit** — A moderate K smooths out noise while still "
                "capturing the real decision boundary."
            )
        else:
            st.warning(
                "🔻 **Underfitting** — Too many neighbors means the model is over-smoothed. "
                "It starts averaging over very different points and loses predictive power."
            )

    # ---- Tab 3: Random Forest n_estimators ----
    with tab3:
        st.subheader("Random Forest: Effect of n_estimators")

        rf_ns, rf_train, rf_test = curves["rf"]

        n_trees = st.slider("Select n_estimators", 10, 300, 100, step=10, key="rf_n")

        fig = overfitting_line_chart(
            rf_ns, rf_train, rf_test,
            "n_estimators", "Random Forest: Training vs Test Accuracy"
        )
        n_idx = rf_ns.index(n_trees)
        fig.add_scatter(
            x=[n_trees], y=[rf_test[n_idx]],
            mode="markers", marker=dict(size=15, color=COLORS["orange"], symbol="star"),
            name=f"Selected (n={n_trees})",
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("Training Accuracy", f"{rf_train[n_idx]:.4f}")
        c2.metric("Test Accuracy", f"{rf_test[n_idx]:.4f}")

        st.info(
            "Random Forest shows **diminishing returns** as the number of trees increases. "
            "After ~100-150 trees, accuracy stabilizes. Adding more trees increases computation "
            "without meaningful improvement — this is why Random Forest is relatively resistant "
            "to overfitting compared to a single Decision Tree."
        )

    # ---- Tab 4: Learning Curves ----
    with tab4:
        st.subheader("Learning Curves (Random Forest)")
        st.markdown(
            "Learning curves show how model performance changes as "
            "the training set size increases."
        )

        if st.button("📊 Compute Learning Curves", type="primary"):
            with st.spinner("Computing learning curves (this involves cross-validation)..."):
                # Need unprocessed encoded data for learning curves
                from utils.preprocessing import encode_features
                df_enc, _ = encode_features(df)
                X_all = df_enc.drop(columns=["Churn"])
                y_all = df_enc["Churn"]
                train_sizes, train_scores, val_scores = get_learning_curve_data(
                    X_all, y_all, 42
                )
                st.session_state["learning_curves"] = (train_sizes, train_scores, val_scores)

        if "learning_curves" in st.session_state:
            train_sizes, train_scores, val_scores = st.session_state["learning_curves"]
            st.plotly_chart(
                learning_curve_chart(train_sizes, train_scores, val_scores),
                use_container_width=True,
            )

            st.info(
                "The **training score** starts high (easy to fit small data) and decreases as "
                "more data is added. The **cross-validation score** increases with more data "
                "and eventually plateaus. When both curves converge, more data won't help — "
                "you need a more complex model."
            )
        else:
            st.info("Click the button above to compute learning curves.")

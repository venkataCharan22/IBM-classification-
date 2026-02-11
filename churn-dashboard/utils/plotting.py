import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Color palette
COLORS = {
    "cyan": "#0EA5E9",
    "teal": "#06B6D4",
    "purple": "#8B5CF6",
    "green": "#10B981",
    "orange": "#F59E0B",
    "red": "#EF4444",
    "navy": "#0F172A",
    "slate": "#1E293B",
    "light": "#F1F5F9",
    "muted": "#94A3B8",
}

CHURN_COLORS = {"No": COLORS["cyan"], "Yes": COLORS["red"]}
CHURN_COLORS_BINARY = {0: COLORS["cyan"], 1: COLORS["red"]}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["light"], size=13, family="Inter, sans-serif"),
    margin=dict(l=50, r=30, t=60, b=50),
    legend=dict(
        bgcolor="rgba(30,41,59,0.7)",
        bordercolor="rgba(14,165,233,0.2)",
        borderwidth=1,
        font=dict(size=12),
    ),
)


def apply_layout(fig, title=None, **kwargs):
    """Apply standard dark-theme layout to a plotly figure."""
    layout = {**PLOTLY_LAYOUT, **kwargs}
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(size=17, color=COLORS["light"], family="Inter, sans-serif"),
            x=0.5, xanchor="center",
        )
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)", zeroline=False,
        tickfont=dict(size=12, color=COLORS["muted"]),
        title_font=dict(size=13, color=COLORS["muted"]),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)", zeroline=False,
        tickfont=dict(size=12, color=COLORS["muted"]),
        title_font=dict(size=13, color=COLORS["muted"]),
    )
    return fig


def churn_distribution_charts(df):
    """Return pie chart and bar chart for churn distribution."""
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]

    fig_pie = px.pie(
        counts, names="Churn", values="Count",
        color="Churn", color_discrete_map=CHURN_COLORS,
        hole=0.45,
    )
    fig_pie.update_traces(
        textinfo="percent+label",
        textfont_size=14,
        marker=dict(line=dict(color=COLORS["navy"], width=2)),
    )
    apply_layout(fig_pie, "Churn Distribution", height=400)

    fig_bar = px.bar(
        counts, x="Churn", y="Count", color="Churn",
        color_discrete_map=CHURN_COLORS, text="Count",
    )
    fig_bar.update_traces(
        textposition="outside", textfont_size=14,
        marker_line_color=COLORS["navy"], marker_line_width=1,
        width=0.5,
    )
    apply_layout(fig_bar, "Churn Count", height=400)
    fig_bar.update_xaxes(title="")
    fig_bar.update_yaxes(title="Number of Customers")

    return fig_pie, fig_bar


def churn_by_contract(df):
    """Stacked bar chart of churn by contract type."""
    ct = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    ct = ct.reset_index().melt(id_vars="Contract", var_name="Churn", value_name="Percentage")

    fig = px.bar(
        ct, x="Contract", y="Percentage", color="Churn",
        color_discrete_map=CHURN_COLORS, barmode="stack",
        text=ct["Percentage"].round(1).astype(str) + "%",
    )
    fig.update_traces(textposition="inside", textfont_size=13)
    apply_layout(fig, "Churn Rate by Contract Type", height=420)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Percentage (%)")
    return fig


def tenure_histogram(df):
    """Histogram of tenure colored by churn."""
    fig = px.histogram(
        df, x="tenure", color="Churn",
        color_discrete_map=CHURN_COLORS,
        nbins=40, barmode="overlay", opacity=0.75,
    )
    fig.add_vline(
        x=12, line_dash="dash", line_color=COLORS["orange"], line_width=2,
        annotation_text="12-month danger zone", annotation_position="top right",
        annotation_font_color=COLORS["orange"], annotation_font_size=12,
    )
    apply_layout(fig, "Tenure Distribution by Churn Status", height=420)
    fig.update_xaxes(title="Tenure (months)")
    fig.update_yaxes(title="Count")
    return fig


def monthly_charges_box(df):
    """Box plot of monthly charges by churn."""
    fig = px.box(
        df, x="Churn", y="MonthlyCharges", color="Churn",
        color_discrete_map=CHURN_COLORS,
    )
    fig.update_traces(marker_line_color=COLORS["navy"], marker_line_width=1)
    apply_layout(fig, "Monthly Charges: Churned vs Retained", height=420)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Monthly Charges ($)")
    return fig


def service_impact_chart(df):
    """Grouped bar chart: churn rate for with/without each service."""
    services = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection"]
    rows = []
    for svc in services:
        if svc not in df.columns:
            continue
        for val in ["Yes", "No"]:
            subset = df[df[svc] == val]
            if len(subset) == 0:
                continue
            churn_rate = (subset["Churn"] == "Yes").mean() * 100
            rows.append({"Service": svc, "Has Service": val, "Churn Rate (%)": churn_rate})

    sdf = pd.DataFrame(rows)
    fig = px.bar(
        sdf, x="Service", y="Churn Rate (%)", color="Has Service",
        barmode="group",
        color_discrete_map={"Yes": COLORS["green"], "No": COLORS["red"]},
        text=sdf["Churn Rate (%)"].round(1).astype(str) + "%",
    )
    fig.update_traces(textposition="outside", textfont_size=12, width=0.35)
    apply_layout(fig, "Service Impact on Churn", height=450)
    fig.update_xaxes(title="")
    return fig


def internet_service_churn(df):
    """Churn rate by internet service type."""
    ct = df.groupby("InternetService")["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    ct.columns = ["InternetService", "Churn Rate (%)"]

    fig = px.bar(
        ct, x="InternetService", y="Churn Rate (%)",
        color="InternetService",
        color_discrete_sequence=[COLORS["cyan"], COLORS["purple"], COLORS["teal"]],
        text=ct["Churn Rate (%)"].round(1).astype(str) + "%",
    )
    fig.update_traces(textposition="outside", textfont_size=13, width=0.5)
    apply_layout(fig, "Churn Rate by Internet Service Type", height=420)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Churn Rate (%)")
    fig.update_layout(showlegend=False)
    return fig


def correlation_heatmap(df_encoded):
    """Full correlation heatmap."""
    corr = df_encoded.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[
            [0.0, "#0F172A"], [0.25, "#164E63"], [0.5, "#1E293B"],
            [0.75, "#06B6D4"], [1.0, "#0EA5E9"],
        ],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9, color=COLORS["light"]),
    ))
    apply_layout(fig, "Feature Correlation Matrix", height=700)
    return fig


def top_churn_predictors(df_encoded):
    """Horizontal bar chart of top 10 features correlated with Churn."""
    if "Churn" not in df_encoded.columns:
        return go.Figure()
    corr = df_encoded.corr()["Churn"].drop("Churn").abs().sort_values(ascending=True)
    top10 = corr.tail(10)

    n = len(top10)
    bar_colors = [
        f"rgba(14, 165, 233, {0.3 + 0.7 * i / (n - 1)})" for i in range(n)
    ]

    fig = go.Figure(go.Bar(
        x=top10.values,
        y=top10.index,
        orientation="h",
        marker_color=bar_colors,
        marker_line_color=COLORS["cyan"],
        marker_line_width=1,
        text=[f"{v:.3f}" for v in top10.values],
        textposition="outside",
        textfont=dict(size=12),
    ))
    apply_layout(fig, "Top 10 Churn Predictors (Absolute Correlation)", height=450)
    fig.update_xaxes(title="Absolute Correlation with Churn", range=[0, top10.max() * 1.2])
    fig.update_yaxes(title="")
    return fig


def class_distribution_bar(dist, title="Class Distribution"):
    """Bar chart from a dict {0: count, 1: count} or {label: count}."""
    label_map = {0: "Not Churned", 1: "Churned", "0": "Not Churned", "1": "Churned"}
    labels = [label_map.get(k, str(k)) for k in dist.keys()]
    values = list(dist.values())
    colors = [COLORS["cyan"], COLORS["red"]] if len(labels) == 2 else [COLORS["cyan"]]

    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{v:,}" for v in values],
        textposition="outside",
        textfont=dict(size=14, color=COLORS["light"]),
        marker_line_color=COLORS["navy"],
        marker_line_width=1,
        width=0.5,
    ))
    apply_layout(fig, title, height=380)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Count")
    fig.update_layout(yaxis_range=[0, max(values) * 1.2])
    return fig


def confusion_matrix_chart(cm, title="Confusion Matrix"):
    """Plotly heatmap for confusion matrix with counts and percentages."""
    total = cm.sum()
    pct = cm / total * 100

    text = [[f"{cm[i][j]:,}<br>({pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)]

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Predicted: No", "Predicted: Yes"],
        y=["Actual: No", "Actual: Yes"],
        colorscale=[
            [0, "#1E293B"], [0.5, "#164E63"], [1, "#0EA5E9"],
        ],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=15, color=COLORS["light"]),
        showscale=False,
        xgap=3, ygap=3,
    ))
    apply_layout(fig, title, height=400)
    return fig


def model_comparison_bar(results):
    """Grouped bar chart comparing all models on key metrics."""
    rows = []
    for name, res in results.items():
        for metric, val in res["metrics"].items():
            rows.append({"Model": name, "Metric": metric.capitalize(), "Score": val})
    mdf = pd.DataFrame(rows)

    color_map = {
        "Accuracy": COLORS["cyan"],
        "Precision": COLORS["teal"],
        "Recall": COLORS["purple"],
        "F1": COLORS["green"],
    }

    fig = px.bar(
        mdf, x="Model", y="Score", color="Metric",
        barmode="group",
        color_discrete_map=color_map,
        text=mdf["Score"].round(3),
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    apply_layout(fig, "Model Comparison", height=500)
    fig.update_xaxes(title="")
    fig.update_yaxes(range=[0, 1.12], title="Score")
    return fig


def f1_ranking_chart(results):
    """Horizontal bar chart ranking models by F1 score."""
    data = [(name, res["metrics"]["f1"]) for name, res in results.items()]
    data.sort(key=lambda x: x[1])
    names, scores = zip(*data)

    n = len(names)
    bar_colors = [COLORS["teal"]] * n
    bar_colors[-1] = COLORS["green"]

    fig = go.Figure(go.Bar(
        x=list(scores), y=list(names), orientation="h",
        marker_color=bar_colors,
        marker_line_color=COLORS["navy"],
        marker_line_width=1,
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        textfont=dict(size=13),
    ))
    apply_layout(fig, "F1 Score Ranking", height=400)
    fig.update_xaxes(range=[0, 1.12], title="F1 Score")
    fig.update_yaxes(title="")
    return fig


def overfitting_line_chart(x_values, train_accs, test_accs, x_label, title):
    """Line chart showing train vs test accuracy for overfitting demo."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values, y=train_accs, mode="lines+markers",
        name="Training Accuracy",
        line=dict(color=COLORS["cyan"], width=2.5),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=x_values, y=test_accs, mode="lines+markers",
        name="Test Accuracy",
        line=dict(color=COLORS["red"], width=2.5),
        marker=dict(size=4),
    ))

    # Shade the gap area
    fig.add_trace(go.Scatter(
        x=list(x_values) + list(x_values)[::-1],
        y=list(train_accs) + list(test_accs)[::-1],
        fill="toself",
        fillcolor="rgba(239, 68, 68, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Overfit Gap",
        showlegend=False,
    ))

    # Mark sweet spot (max test accuracy)
    best_idx = int(np.argmax(test_accs))
    fig.add_vline(
        x=x_values[best_idx], line_dash="dash", line_color=COLORS["green"], line_width=1.5,
        annotation_text=f"Best: {x_label}={x_values[best_idx]}",
        annotation_position="top right",
        annotation_font_color=COLORS["green"],
        annotation_font_size=12,
    )

    apply_layout(fig, title, height=450)
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title="Accuracy", range=[0.5, 1.02])
    return fig


def learning_curve_chart(train_sizes, train_scores, val_scores):
    """Learning curve plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_scores, mode="lines+markers",
        name="Training Score",
        line=dict(color=COLORS["cyan"], width=2.5),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_scores, mode="lines+markers",
        name="Cross-Validation Score",
        line=dict(color=COLORS["purple"], width=2.5),
        marker=dict(size=6),
    ))
    apply_layout(fig, "Learning Curves (Random Forest)", height=450)
    fig.update_xaxes(title="Training Set Size")
    fig.update_yaxes(title="Accuracy", range=[0.5, 1.02])
    return fig


def cv_fold_scores_chart(scores, n_folds):
    """Bar chart of individual fold scores."""
    fold_labels = [f"Fold {i+1}" for i in range(n_folds)]

    fig = go.Figure(go.Bar(
        x=fold_labels, y=scores,
        marker_color=COLORS["cyan"],
        marker_line_color=COLORS["navy"],
        marker_line_width=1,
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
        textfont=dict(size=12),
        width=0.6,
    ))
    fig.add_hline(
        y=scores.mean(), line_dash="dash", line_color=COLORS["green"], line_width=2,
        annotation_text=f"Mean: {scores.mean():.4f}",
        annotation_position="top right",
        annotation_font_color=COLORS["green"],
        annotation_font_size=13,
    )
    apply_layout(fig, f"{n_folds}-Fold Cross-Validation Scores", height=420)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Score")
    return fig


def feature_distribution_chart(values_unscaled, values_scaled, feature_name):
    """Side-by-side histograms showing scaled vs unscaled distribution."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Original (Unscaled)", "After StandardScaler"],
    )

    fig.add_trace(
        go.Histogram(
            x=values_unscaled, marker_color=COLORS["cyan"],
            opacity=0.75, name="Unscaled",
            marker_line_color=COLORS["navy"], marker_line_width=1,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=values_scaled, marker_color=COLORS["purple"],
            opacity=0.75, name="Scaled",
            marker_line_color=COLORS["navy"], marker_line_width=1,
        ),
        row=1, col=2,
    )

    apply_layout(fig, f"{feature_name}: Unscaled vs Scaled", height=380)
    fig.update_annotations(font=dict(size=14, color=COLORS["light"]))
    return fig

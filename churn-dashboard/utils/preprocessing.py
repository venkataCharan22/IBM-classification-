import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE


def encode_features(df):
    """Label-encode all categorical columns. Returns encoded df and encoder mappings."""
    df_encoded = df.copy()
    label_encoders = {}

    for col in df_encoded.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    return df_encoded, label_encoders


def prepare_data(df, feature_groups, test_size=0.2, use_smote=False,
                 use_scaling=False, random_state=42):
    """
    Full preprocessing pipeline.

    Returns dict with X_train, X_test, y_train, y_test,
    feature_names, class_dist_before, class_dist_after, scaler.
    """
    from utils.data_loader import get_feature_groups

    all_groups = get_feature_groups()
    selected_features = []
    for group in feature_groups:
        selected_features.extend(all_groups[group])

    # Keep only features that exist in df
    selected_features = [f for f in selected_features if f in df.columns]

    if not selected_features:
        return None

    # Encode
    df_encoded, _ = encode_features(df[selected_features + ["Churn"]])

    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    # Class distribution before
    class_dist_before = y.value_counts().to_dict()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # SMOTE — only on training data
    class_dist_after = y_train.value_counts().to_dict()
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        class_dist_after = pd.Series(y_train).value_counts().to_dict()

    # Scaling
    scaler = None
    if use_scaling:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=X.columns
        )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "class_dist_before": class_dist_before,
        "class_dist_after": class_dist_after,
        "scaler": scaler,
    }

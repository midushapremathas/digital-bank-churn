import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


df = pd.read_csv("data/processed/cleaned_churn_data.csv")

df["last_active_date"] = pd.to_datetime(df["last_active_date"])
df["created_date"] = pd.to_datetime(df["created_date"])

df["last_active_year"] = df["last_active_date"].dt.year
df["last_active_month"] = df["last_active_date"].dt.month
df["created_year"] = df["created_date"].dt.year
df["created_month"] = df["created_date"].dt.month

df = df.drop(columns=["last_active_date", "created_date"])

X = df.drop(columns=["exit"])
y = df["exit"]

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

y_probability = model.predict_proba(X_test)[:, 1]

print("Gradient Boosting ROC-AUC:")
print(roc_auc_score(y_test, y_probability))

print("\nThreshold analysis:")

print("\nDetailed threshold analysis:")

results = []

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:

    y_pred_threshold = (y_probability >= threshold).astype(int)

    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)

    results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

results_df = pd.DataFrame(results)

print(results_df.round(3).to_string(index=False))

risk_segments = pd.cut(
    y_probability,
    bins=[0, 0.25, 0.50, 0.75, 1.00],
    labels=["Low", "Moderate", "High", "Critical"],
    include_lowest=True
)

print("\nCustomer risk segmentation:")
print(risk_segments.value_counts().sort_index())
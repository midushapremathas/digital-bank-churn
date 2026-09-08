import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score


# Load the cleaned dataset
df = pd.read_csv("data/processed/cleaned_churn_data.csv")

# Convert date columns
df["last_active_date"] = pd.to_datetime(df["last_active_date"])
df["created_date"] = pd.to_datetime(df["created_date"])

# Create numerical date features
df["last_active_year"] = df["last_active_date"].dt.year
df["last_active_month"] = df["last_active_date"].dt.month
df["created_year"] = df["created_date"].dt.year
df["created_month"] = df["created_date"].dt.month

# Remove original date columns
df = df.drop(columns=["last_active_date", "created_date"])

# Separate target and predictors
X = df.drop(columns=["exit"])
y = df["exit"]

# Identify categorical features
categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns

# Preprocessing
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

# Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))

print("\nRandom Forest ROC-AUC:")
print(roc_auc_score(y_test, y_probability))
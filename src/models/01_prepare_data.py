import pandas as pd

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

print("Modelling dataset shape:")
print(X.shape)

print("\nModelling features:")
print(X.columns.tolist())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:")
print(X_train.shape)

print("\nTest set shape:")
print(X_test.shape)

print("\nTraining churn rate:")
print(y_train.mean())

print("\nTest churn rate:")
print(y_test.mean())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = X.select_dtypes(include=["object", "str"]).columns
numerical_features = X.select_dtypes(exclude=["object"]).columns

print("\nCategorical features:")
print(categorical_features.tolist())

print("\nNumerical features:")
print(numerical_features.tolist())

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

print("\nPreprocessor created successfully.")

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training data shape:")
print(X_train_processed.shape)

print("\nProcessed test data shape:")
print(X_test_processed.shape)
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/processed/cleaned_churn_data.csv")

# Convert date columns to datetime
df["last_active_date"] = pd.to_datetime(df["last_active_date"])
df["created_date"] = pd.to_datetime(df["created_date"])

# Create useful numerical date features
df["last_active_year"] = df["last_active_date"].dt.year
df["last_active_month"] = df["last_active_date"].dt.month

df["created_year"] = df["created_date"].dt.year
df["created_month"] = df["created_date"].dt.month

# Drop the original date columns
df = df.drop(columns=["last_active_date", "created_date"])

# Separate target from predictors
X = df.drop(columns=["exit"])
y = df["exit"]

print("Modelling dataset shape:")
print(X.shape)

print("\nModelling features:")
print(X.columns.tolist())

from sklearn.model_selection import train_test_split

# Split data into training and test sets
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

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=["object", "str"]).columns
numerical_features = X.select_dtypes(exclude=["object"]).columns

print("\nCategorical features:")
print(categorical_features.tolist())

print("\nNumerical features:")
print(numerical_features.tolist())

# Create preprocessing pipeline
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

# Fit the preprocessor on the training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform the test data using the fitted preprocessor
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training data shape:")
print(X_train_processed.shape)

print("\nProcessed test data shape:")
print(X_test_processed.shape)
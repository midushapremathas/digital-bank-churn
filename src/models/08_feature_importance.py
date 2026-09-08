import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline


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

# Gradient Boosting model
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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train the model
model.fit(X_train, y_train)

# Get feature names after preprocessing
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

# Get feature importance
importance = model.named_steps[
    "classifier"
].feature_importances_

# Create feature importance table
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Sort by importance
feature_importance_df = feature_importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nTop 20 features influencing churn predictions:")
print(feature_importance_df.head(20).to_string(index=False))
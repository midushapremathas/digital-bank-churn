import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/processed/cleaned_churn_data.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)

# Create a simple risk profile using the model's predicted probabilities
print("\nPreparing risk profile analysis...")

# Separate features and target
X = df.drop(columns=["exit"])
y = df["exit"]

print("\nFeatures and target prepared.")
print("Features:", X.shape)
print("Target:", y.shape)

from sklearn.model_selection import train_test_split

# Split the data using the same approach as our main model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData split complete.")
print("Training data:", X_train.shape)
print("Test data:", X_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Identify categorical and numerical features
categorical_features = [
    "gender",
    "occupation",
    "address",
    "origin_province",
    "customer_segment",
    "loyalty_level",
    "digital_behavior"
]

numerical_features = [
    "credit_sco",
    "age",
    "balance",
    "monthly_ir",
    "tenure_ye",
    "married",
    "nums_card",
    "nums_service",
    "active_member"
]

# One-hot encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", "passthrough", numerical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing complete.")
print("Processed training data:", X_train_processed.shape)

from sklearn.ensemble import GradientBoostingClassifier

# Train the Gradient Boosting model
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

model.fit(X_train_processed, y_train)

print("\nGradient Boosting model trained successfully.")

# Predict churn probabilities
y_probability = model.predict_proba(X_test_processed)[:, 1]

print("\nChurn probabilities generated.")
print("Highest predicted risk:", y_probability.max())

# Create risk segments for the test customers
risk_segments = pd.cut(
    y_probability,
    bins=[0, 0.25, 0.50, 0.75, 1.00],
    labels=["Low", "Moderate", "High", "Critical"],
    include_lowest=True
)

print("\nRisk segment counts:")
print(risk_segments.value_counts().sort_index())

# Add predictions and risk segments to the test customer data
risk_profile = X_test.copy()
risk_profile["predicted_churn_probability"] = y_probability
risk_profile["risk_segment"] = risk_segments

print("\nRisk profile created.")
print(risk_profile[["predicted_churn_probability", "risk_segment"]].head())

# Analyse the characteristics of high-risk customers
high_risk = risk_profile[
    risk_profile["risk_segment"].isin(["High", "Critical"])
]

print("\nHigh-risk customer profile:")
print("Number of high-risk customers:", len(high_risk))

print("\nAverage characteristics:")
print(
    high_risk[
        ["age", "balance", "monthly_ir", "tenure_ye",
         "nums_card", "nums_service", "credit_sco"]
    ].mean()
)

# Analyse categorical characteristics of high-risk customers

print("\nHigh-risk customer categorical profile:")

print("\nActive membership:")
print(high_risk["active_member"].value_counts(normalize=True))

print("\nCustomer segment:")
print(high_risk["customer_segment"].value_counts(normalize=True))

print("\nLoyalty level:")
print(high_risk["loyalty_level"].value_counts(normalize=True))

print("\nDigital behaviour:")
print(high_risk["digital_behavior"].value_counts(normalize=True))

print("\nNumber of services:")
print(high_risk["nums_service"].value_counts(normalize=True).sort_index())

# Compare high-risk customers with the rest of the test population

other_customers = risk_profile[
    risk_profile["risk_segment"].isin(["Low", "Moderate"])
]

print("\nComparison: high-risk vs lower-risk customers")

print("\nActive membership:")
print(pd.DataFrame({
    "High-risk": high_risk["active_member"].value_counts(normalize=True),
    "Lower-risk": other_customers["active_member"].value_counts(normalize=True)
}))

print("\nCustomer segment:")
print(pd.DataFrame({
    "High-risk": high_risk["customer_segment"].value_counts(normalize=True),
    "Lower-risk": other_customers["customer_segment"].value_counts(normalize=True)
}))

print("\nLoyalty level:")
print(pd.DataFrame({
    "High-risk": high_risk["loyalty_level"].value_counts(normalize=True),
    "Lower-risk": other_customers["loyalty_level"].value_counts(normalize=True)
}))

print("\nDigital behaviour:")
print(pd.DataFrame({
    "High-risk": high_risk["digital_behavior"].value_counts(normalize=True),
    "Lower-risk": other_customers["digital_behavior"].value_counts(normalize=True)
}))

# Evaluate observed churn rate within each predicted risk segment

risk_profile["actual_churn"] = y_test.values

print("\nObserved churn rate by predicted risk segment:")

print(
    risk_profile.groupby("risk_segment")["actual_churn"].mean()
)

# Calculate customer value at risk

high_risk_balance = high_risk["balance"].sum()
high_risk_income = high_risk["monthly_ir"].sum()

print("\nCustomer value at risk:")

print("Total balance held by high-risk customers:",
      high_risk_balance)

print("Total monthly income associated with high-risk customers:",
      high_risk_income)

# Calculate the proportion of total customer balance at risk

total_balance = risk_profile["balance"].sum()

balance_at_risk_percentage = (
    high_risk_balance / total_balance
) * 100

print("\nProportion of customer balance associated with high-risk customers:")
print(f"{balance_at_risk_percentage:.2f}%")

# Calculate customer value associated with critical-risk customers

critical_risk = risk_profile[
    risk_profile["risk_segment"] == "Critical"
]

critical_balance = critical_risk["balance"].sum()

critical_balance_percentage = (
    critical_balance / total_balance
) * 100

print("\nCritical-risk customer value:")
print("Number of critical-risk customers:", len(critical_risk))
print("Total balance held by critical-risk customers:", critical_balance)
print(
    f"Proportion of total customer balance: "
    f"{critical_balance_percentage:.2f}%"
)
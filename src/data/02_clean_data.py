import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/raw/bank_churn_dataset.csv")

# Convert date columns to datetime
df["last_active_date"] = pd.to_datetime(
    df["last_active_date"],
    format="%d/%m/%Y"
)

df["created_date"] = pd.to_datetime(
    df["created_date"],
    format="%d/%m/%Y"
)

# Remove columns that are identifiers, personal information,
# pre-existing risk metrics, or unreliable fields
columns_to_drop = [
    "id",
    "full_name",
    "engagement_score",
    "risk_score",
    "risk_segment",
    "cluster_group",
    "last_transaction_month"
]

df_clean = df.drop(columns=columns_to_drop)

# Save the cleaned dataset
df_clean.to_csv(
    "data/processed/cleaned_churn_data.csv",
    index=False
)

print("Cleaned dataset created successfully.")
print("\nCleaned dataset shape:")
print(df_clean.shape)

print("\nRemaining columns:")
print(df_clean.columns.tolist())

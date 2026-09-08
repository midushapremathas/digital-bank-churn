import pandas as pd

# Load the raw dataset
df = pd.read_csv("data/raw/bank_churn_dataset.csv")

# Show the first 5 rows
print(df.head())

# Show dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Show all column names
print("\nColumn names:")
print(df.columns.tolist())

# Show data types and non-null counts
print("\nData information:")
print(df.info())

# Check for duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Calculate churn rate
print("\nChurn rate:")
print(df["exit"].value_counts(normalize=True))

# Check whether customer IDs are unique
print("\nUnique customer IDs:")
print(df["id"].nunique())

# Check unique values in categorical columns
categorical_columns = df.select_dtypes(include="str").columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].unique())

  # Investigate potential target leakage
suspicious_columns = [
    "engagement_score",
    "risk_score",
    "risk_segment",
    "cluster_group"
]

print("\nPotential leakage investigation:")

for column in suspicious_columns:
    print(f"\n{column} by churn status:")
    
    if df[column].dtype == "str":
        print(pd.crosstab(df["exit"], df[column], normalize="index"))
    else:
        print(df.groupby("exit")[column].agg(["mean", "min", "max"])) 

        # Check relationships between suspicious variables and other features
print("\nSuspicious variable correlations:")

numeric_columns = df.select_dtypes(include=["int64", "float64", "bool"]).columns

correlations = df[numeric_columns].corr()

for column in ["engagement_score", "risk_score", "cluster_group"]:
    print(f"\n{column}:")
    print(correlations[column].sort_values(ascending=False))

    # Investigate how engagement_score is constructed
print("\nEngagement score by active member:")
print(df.groupby("active_member")["engagement_score"].agg(["mean", "min", "max"]))

print("\nEngagement score by number of services:")
print(df.groupby("nums_service")["engagement_score"].mean())

print("\nEngagement score by last transaction month:")
print(df.groupby("last_transaction_month")["engagement_score"].mean())

# Investigate how risk_score is constructed
print("\nRisk score by credit score:")
print(df.groupby(pd.cut(df["credit_sco"], bins=5))["risk_score"].mean())

print("\nRisk score by active member:")
print(df.groupby("active_member")["risk_score"].agg(["mean", "min", "max"]))

print("\nRisk score by number of services:")
print(df.groupby("nums_service")["risk_score"].mean())

print("\nRisk score by monthly income:")
print(df.groupby(pd.qcut(df["monthly_ir"], 5))["risk_score"].mean())

# Check whether risk_score is derived from other variables
print("\nRisk score relationship with key variables:")

risk_check = df[
    ["risk_score", "credit_sco", "monthly_ir", "balance",
     "active_member", "nums_service", "nums_card",
     "tenure_ye", "exit"]
].corr()

print(risk_check["risk_score"].sort_values(ascending=False))

# Check churn rate across risk score groups
print("\nChurn rate by risk score group:")

df["risk_score_group"] = pd.qcut(
    df["risk_score"],
    q=5,
    duplicates="drop"
)

print(
    df.groupby("risk_score_group", observed=True)["exit"]
    .agg(["mean", "count"])
)

# Investigate last_transaction_month
print("\nLast transaction month summary:")
print(df["last_transaction_month"].describe())

print("\nSmallest last_transaction_month values:")
print(sorted(df["last_transaction_month"].unique())[:30])

print("\nLargest last_transaction_month values:")
print(sorted(df["last_transaction_month"].unique())[-30:])

# Investigate relationship between transaction field and last active date
print("\nLast active date range:")
print(df["last_active_date"].min())
print(df["last_active_date"].max())

print("\nLast transaction month by churn status:")
print(
    df.groupby("exit")["last_transaction_month"]
    .agg(["mean", "median", "min", "max"])
)

print("\nProportion with zero transaction month:")
print(
    (df["last_transaction_month"] == 0)
    .mean()
)

# Create a proper datetime version of last active date
df["last_active_date"] = pd.to_datetime(
    df["last_active_date"],
    format="%d/%m/%Y"
)

print("\nLast active date by churn status:")
print(
    df.groupby("exit")["last_active_date"]
    .agg(["min", "max"])
)

print("\nLast active year by churn status:")
print(
    df.groupby(
        [df["last_active_date"].dt.year, "exit"]
    ).size()
)

# Check whether last active date occurs before account creation
df["created_date"] = pd.to_datetime(
    df["created_date"],
    format="%d/%m/%Y"
)

invalid_date_order = df["last_active_date"] < df["created_date"]

print("\nCustomers with last active date before account creation:")
print(invalid_date_order.sum())

print("\nPercentage with invalid date order:")
print(invalid_date_order.mean())
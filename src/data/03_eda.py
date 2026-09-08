import pandas as pd

df = pd.read_csv("data/processed/cleaned_churn_data.csv")

print("Overall churn rate:")
print(df["exit"].value_counts(normalize=True))

print("\nChurn rate by active membership:")
print(df.groupby("active_member")["exit"].mean())

print("\nChurn rate by number of services:")
print(df.groupby("nums_service")["exit"].mean())

print("\nChurn rate by customer segment:")
print(df.groupby("customer_segment")["exit"].mean())

print("\nChurn rate by loyalty level:")
print(df.groupby("loyalty_level")["exit"].mean())

print("\nChurn rate by digital behaviour:")
print(df.groupby("digital_behavior")["exit"].mean())

print("\nChurn rate by gender:")
print(df.groupby("gender")["exit"].mean())

print("\nFinancial characteristics by churn status:")
print(
    df.groupby("exit")[["balance", "monthly_ir"]].mean()
)

print("\nAge and tenure by churn status:")
print(
    df.groupby("exit")[["age", "tenure_ye"]].mean()
)

print("\nBalance summary:")
print(df["balance"].describe())

print("\nMonthly income summary:")
print(df["monthly_ir"].describe())

print("\nTop 10 balance values:")
print(df["balance"].nlargest(10))

print("\nTop 10 monthly income values:")
print(df["monthly_ir"].nlargest(10))

print("\nCustomers with balance of £1 billion:")
print((df["balance"] == 1_000_000_000).sum())

print("\nCustomers with monthly income above £100 million:")
print((df["monthly_ir"] > 100_000_000).sum())

print("\nChurn rate for customers with balance of £1 billion:")
print(
    df.loc[df["balance"] == 1_000_000_000, "exit"].mean()
)

print("\nChurn rate for customers with monthly income above £100 million:")
print(
    df.loc[df["monthly_ir"] > 100_000_000, "exit"].mean()
)

print("\nCustomer segments among customers with monthly income above £100 million:")
print(
    df.loc[df["monthly_ir"] > 100_000_000, "customer_segment"]
    .value_counts()
)

print("\nCustomer segments among customers with £1 billion balance:")
print(
    df.loc[df["balance"] == 1_000_000_000, "customer_segment"]
    .value_counts()
)

print("\nNumerical variable ranges:")

for column in ["age", "balance", "monthly_ir", "tenure_ye", "nums_card", "nums_service"]:
    print(f"\n{column}:")
    print("Minimum:", df[column].min())
    print("Maximum:", df[column].max())

df["age_group"] = pd.cut(
    df["age"],
    bins=[19, 29, 39, 49, 59, 69, 79, 90],
    labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-90"]
)

print("\nChurn rate by age group:")
print(df.groupby("age_group", observed=True)["exit"].mean())

print("\nChurn rate by tenure:")
print(
    df.groupby("tenure_ye")["exit"].mean()
)

print("\nNumerical correlations:")

numeric_columns = [
    "credit_sco",
    "age",
    "balance",
    "monthly_ir",
    "tenure_ye",
    "nums_card",
    "nums_service"
]

print(df[numeric_columns + ["exit"]].corr()["exit"].sort_values(ascending=False))

from scipy.stats import chi2_contingency

print("\nChi-square test: active membership vs churn")

contingency_table = pd.crosstab(
    df["active_member"],
    df["exit"]
)

chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2)
print("p-value:", p_value)

print("\nChi-square test: number of services vs churn")

contingency_table = pd.crosstab(
    df["nums_service"],
    df["exit"]
)

chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2)
print("p-value:", p_value)

print("\nChi-square test: customer segment vs churn")

contingency_table = pd.crosstab(
    df["customer_segment"],
    df["exit"]
)

chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2)
print("p-value:", p_value)

from scipy.stats import mannwhitneyu

print("\nMann-Whitney U test: balance vs churn")

balance_retained = df.loc[df["exit"] == False, "balance"]
balance_churned = df.loc[df["exit"] == True, "balance"]

statistic, p_value = mannwhitneyu(
    balance_retained,
    balance_churned,
    alternative="two-sided"
)

print("U statistic:", statistic)
print("p-value:", p_value)

print("\nMann-Whitney U test: monthly income vs churn")

income_retained = df.loc[df["exit"] == False, "monthly_ir"]
income_churned = df.loc[df["exit"] == True, "monthly_ir"]

statistic, p_value = mannwhitneyu(
    income_retained,
    income_churned,
    alternative="two-sided"
)

print("U statistic:", statistic)
print("p-value:", p_value)

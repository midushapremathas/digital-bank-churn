import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/processed/cleaned_churn_data.csv")



total_customers = len(df)
churned_customers = df["exit"].sum()
churn_rate = df["exit"].mean() * 100

print("Digital Bank Customer Churn Dashboard")
print("=" * 45)

print(f"Total customers: {total_customers:,}")
print(f"Churned customers: {churned_customers:,}")
print(f"Overall churn rate: {churn_rate:.2f}%")



segment_churn = (
    df.groupby("customer_segment")["exit"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\nChurn rate by customer segment:")
print(segment_churn)



services_churn = (
    df.groupby("nums_service")["exit"]
    .mean()
    .mul(100)
)

print("\nChurn rate by number of services:")
print(services_churn)



loyalty_churn = (
    df.groupby("loyalty_level")["exit"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\nChurn rate by loyalty level:")
print(loyalty_churn)



digital_churn = (
    df.groupby("digital_behavior")["exit"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\nChurn rate by digital behaviour:")
print(digital_churn)



plt.figure(figsize=(10, 6))

segment_churn.plot(kind="bar")

plt.title("Customer Churn Rate by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "reports/churn_by_customer_segment.png",
    dpi=300
)

plt.close()



plt.figure(figsize=(10, 6))

services_churn.plot(kind="line", marker="o")

plt.title("Customer Churn Rate by Number of Services")
plt.xlabel("Number of Services")
plt.ylabel("Churn Rate (%)")
plt.xticks(services_churn.index)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "reports/churn_by_number_of_services.png",
    dpi=300
)

plt.close()



plt.figure(figsize=(10, 6))

loyalty_churn.plot(kind="bar")

plt.title("Customer Churn Rate by Loyalty Level")
plt.xlabel("Loyalty Level")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "reports/churn_by_loyalty_level.png",
    dpi=300
)

plt.close()



plt.figure(figsize=(10, 6))

digital_churn.plot(kind="bar")

plt.title("Customer Churn Rate by Digital Behaviour")
plt.xlabel("Digital Behaviour")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    "reports/churn_by_digital_behaviour.png",
    dpi=300
)

plt.close()



plt.figure(figsize=(10, 6))

plt.hist(
    df["age"],
    bins=15,
    edgecolor="black"
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    "reports/customer_age_distribution.png",
    dpi=300
)

plt.close()


print("\nDashboard visualisations saved successfully.")
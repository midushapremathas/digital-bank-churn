print("Prediction vs Causation")
print("=" * 40)

print("""
What the model can tell us:
- Which customers have a higher predicted probability of churn.
- Which customer characteristics are strongly associated with churn.
- Which groups should be prioritised for further investigation.
- Which features contribute most to the model's predictions.

What the model cannot tell us:
- Whether a specific characteristic directly causes churn.
- Whether a retention intervention will prevent a customer from leaving.
- Whether changing a customer's behaviour would necessarily reduce
  their probability of churn.

Examples:
- Inactivity is strongly associated with churn, but we cannot conclude
  that inactivity directly causes customers to leave.
- Customers with fewer services have higher churn risk, but we cannot
  conclude that adding more services will prevent churn.
- Bronze loyalty customers are overrepresented among high-risk
  customers, but we cannot conclude that Bronze status causes churn.

How this could be investigated:
A future analysis could use controlled experiments such as A/B testing
to measure whether specific retention interventions actually reduce
customer churn.

Conclusion:
The model should be used as a decision-support and customer
prioritisation tool rather than as evidence of causal relationships.
""")
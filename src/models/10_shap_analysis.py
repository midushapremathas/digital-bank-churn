import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline


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

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns

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

classifier = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

classifier.fit(X_train_processed, y_train)

explainer = shap.TreeExplainer(classifier)

X_sample = X_test_processed[:1000]

shap_values = explainer.shap_values(X_sample)

mean_shap = abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_shap
})

shap_importance = shap_importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

print("\nTop 20 features by SHAP importance:")
print(
    shap_importance.head(20).to_string(index=False)
)

y_probability = classifier.predict_proba(X_sample)[:, 1]

high_risk_index = y_probability.argmax()

print("\nSelected customer:")
print(
    "Predicted churn probability:",
    y_probability[high_risk_index]
)

individual_values = shap_values[high_risk_index]

individual_data = X_sample[high_risk_index]

if hasattr(individual_data, "toarray"):
    individual_data = individual_data.toarray().flatten()
else:
    individual_data = individual_data.flatten()

print("\nSHAP expected value:")
print(explainer.expected_value)
print("Expected value shape:", np.asarray(explainer.expected_value).shape)

print("\nIndividual SHAP shape:")
print(individual_values.shape)
individual_shap = shap.Explanation(
    values=individual_values,
    base_values=explainer.expected_value[0],
    data=individual_data,
    feature_names=feature_names
)

shap.plots.waterfall(
    individual_shap,
    max_display=15
)

plt.savefig("reports/shap_waterfall.png", bbox_inches="tight")
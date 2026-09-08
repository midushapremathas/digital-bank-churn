# Digital Bank Customer Churn Intelligence System

## Project Overview

Customer churn is a major business problem for digital banks. If a bank can identify customers who are more likely to leave, it can prioritise them for retention strategies and investigate the behaviours associated with customer loss.

This project develops an end-to-end customer churn intelligence system using a simulated dataset of 80,000 digital banking customers.

The aim is to answer five key questions:

* Which customers are most likely to churn?
* Which customer characteristics are associated with churn?
* Which factors contribute most strongly to model predictions?
* How can customers be segmented according to their predicted risk?
* How could the analysis support targeted retention strategies?

The project combines **data cleaning, exploratory analysis, statistical testing, SQL, machine learning, model evaluation, explainable AI and business analysis**.

---

## Dataset

The dataset contains 80,000 simulated bank customer records and includes information relating to:

* Customer demographics
* Credit scores
* Account balances
* Monthly income
* Customer segment
* Loyalty level
* Number of services
* Digital behaviour
* Active membership
* Customer tenure
* Churn status

Several pre-existing derived variables were investigated for potential data leakage. Variables such as risk score, risk segment, engagement score and cluster group were excluded from the main predictive model where appropriate.

The dataset is simulated, so financial values should not be interpreted as real-world banking figures.

---

## Data Preparation

The raw dataset was audited for:

* Missing values
* Duplicate records
* Invalid dates
* Extreme values
* Unique identifiers
* Potential target leakage
* Derived variables
* Implausible or corrupted fields

The dataset contained no missing values or duplicate records.

The `last_transaction_month` variable was identified as unreliable because of highly abnormal values and was excluded.

The final modelling dataset contains 19 variables after removing identifiers, corrupted fields and potentially leaky derived variables.

---

## Exploratory Data Analysis

Initial analysis found substantial differences in churn between customer groups.

Overall churn rate:

**18.0%**

Examples of observed churn rates:

| Factor | Group | Churn Rate |
|---|---|---:|
| Customer segment | Mass | 39.48% |
| Customer segment | Emerging | 17.62% |
| Customer segment | Affluent | 3.10% |
| Customer segment | Priority | 0.07% |
| Digital behaviour | Offline | 21.14% |
| Digital behaviour | Mobile | 6.50% |
| Loyalty | Bronze | 20.24% |
| Loyalty | Silver | 3.68% |
| Loyalty | Gold | 2.82% |

Churn also decreased substantially as the number of services held by a customer increased.

These relationships were investigated further using statistical tests rather than relying solely on visual patterns.

---

## Statistical Analysis

Several statistical tests were performed to determine whether observed differences were statistically significant.

Examples include:

* Chi-square tests for categorical variables
* Mann–Whitney U tests for heavily skewed numerical variables
* Correlation analysis
* Group-level churn comparisons

There was strong statistical evidence of associations between churn and:

* Active membership
* Number of services
* Customer segment
* Loyalty level
* Digital behaviour
* Balance
* Monthly income

Statistical association does **not** imply causation. The analysis therefore distinguishes between predictive relationships and causal conclusions.

---

## Machine Learning

Four classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

Because churn represents a minority class, class imbalance was considered when training and evaluating models.

### Model comparison

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.348 | 0.836 | 0.492 | 0.822 | 0.467 |
| Decision Tree | 0.382 | 0.811 | 0.520 | 0.840 | 0.482 |
| Random Forest | 0.384 | 0.851 | 0.529 | 0.851 | 0.528 |
| Gradient Boosting | 0.619 | 0.335 | 0.435 | 0.856 | 0.541 |

Gradient Boosting achieved the strongest ROC-AUC and PR-AUC.

However, its default probability threshold produced relatively low recall. Threshold analysis was therefore performed to investigate the trade-off between identifying more potential churners and increasing false positives.

At a threshold of **0.25**, Gradient Boosting achieved approximately:

* Precision: 0.442
* Recall: 0.721
* F1: 0.548

This demonstrates why classification thresholds should be considered in the context of the business objective rather than automatically using 0.50.

---

## Model Explainability

Feature importance and SHAP analysis were used to investigate which variables contributed most strongly to the Gradient Boosting model's predictions.

The strongest model influences included:

* Monthly income
* Balance
* Credit score
* Number of services
* Digital behaviour
* Customer segment
* Tenure
* Active membership

SHAP was also used to examine individual predictions and identify which features pushed a customer's predicted churn probability higher or lower.

SHAP explanations describe model behaviour and should not be interpreted as evidence that a variable directly causes churn.

---

## Customer Risk Segmentation

Customers were divided into four predicted risk categories using the Gradient Boosting model:

* **Low risk**
* **Moderate risk**
* **High risk**
* **Critical risk**

The high and critical risk groups contained approximately **1,558 customers** in the analysed test population.

High-risk customers were overwhelmingly characterised by:

* Inactive membership
* Offline digital behaviour
* Mass customer segment
* Bronze loyalty
* Relatively low numbers of services

The observed churn rate increased substantially across the predicted risk groups, providing evidence that the model can be useful for customer prioritisation.

---

## Business Recommendations

The analysis suggests several areas for further investigation by a digital bank:

### 1. Re-engage inactive customers

Inactive membership was strongly associated with higher churn risk. Customers showing signs of declining engagement could be prioritised for re-engagement campaigns.

### 2. Increase digital engagement

Offline customers showed substantially higher churn than mobile customers. Digital onboarding, personalised app communications and improved digital engagement could be investigated as potential retention strategies.

### 3. Prioritise Mass-segment customers

Mass customers represented a particularly high-risk group and could be considered a priority for targeted retention analysis.

### 4. Investigate low-product adoption

Customers with fewer services had substantially higher churn rates. The bank could investigate whether appropriate cross-selling or product education could improve retention.

### 5. Use risk scores for prioritisation

Rather than treating every customer equally, the model could help the bank prioritise customers for further investigation based on predicted churn probability.

These recommendations are hypotheses for business action, not claims of causality.

---

## SQL Analysis

SQL was used to reproduce and extend key business analyses, including:

* Overall churn
* Churn by customer segment
* Churn by number of services
* Churn by active membership
* Churn by loyalty level
* Churn by digital behaviour
* Financial characteristics by churn status
* High-risk customer groups
* Customer balance potentially exposed to churn

The SQL analysis demonstrates the ability to translate data science findings into business-focused database queries.

---

## Project Structure

```text
digital-bank-churn/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│   ├── feature_importance.png
│   ├── shap_waterfall.png
│   ├── churn_by_customer_segment.png
│   ├── churn_by_number_of_services.png
│   ├── churn_by_loyalty_level.png
│   ├── churn_by_digital_behaviour.png
│   └── customer_age_distribution.png
│
├── sql/
│   └── churn_analysis.sql
│
├── src/
│   ├── data/
│   ├── models/
│   └── dashboard/
│
├── README.md
└── requirements.txt
```

---

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* SHAP
* SQLite
* SQL
* Git / GitHub

---

## Key Skills Demonstrated

**Data Science**

* Data cleaning
* Exploratory data analysis
* Statistical testing
* Feature engineering
* Classification
* Model evaluation
* Threshold optimisation
* Model explainability

**Technical**

* Python
* Pandas
* Scikit-learn
* SHAP
* SQL
* SQLite
* Data visualisation

**Business**

* Customer segmentation
* Churn analysis
* Risk prioritisation
* Retention strategy
* Decision-support analysis
* Distinguishing prediction from causation

---

## Limitations

This project uses a simulated dataset, so the results should not be treated as evidence about real banking customers.

The model identifies patterns associated with churn but does not establish causal relationships.

A production system would require additional considerations such as:

* Real customer behaviour data
* Model monitoring
* Data drift detection
* Fairness and bias assessment
* Privacy and governance
* Cost-sensitive threshold selection
* Controlled retention experiments

---

## Conclusion

This project demonstrates an end-to-end approach to customer churn analysis, combining statistical reasoning, machine learning, explainability, SQL and business decision-making.

Rather than simply predicting whether a customer will churn, the project focuses on **why customers are being flagged, how risk can be prioritised, and how predictive analytics could support practical business decisions**.

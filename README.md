# Bank Customer Churn Prediction

Predicting which bank customers are likely to leave, so the bank can target them with retention offers before they churn. The project covers the full cycle: exploratory analysis, data leakage detection, model comparison, threshold selection and business conclusions.

## Dataset

[Bank Customer Churn](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn) — 10,000 customers, 18 columns: demographics, account balance, number of products, activity status, satisfaction score, card type and the target `Exited` (1 = left the bank).

About 20% of customers churned, so the classes are imbalanced.

## Key findings from the data

- **Number of products** has a non-linear effect: customers with 2 products churn least, while those with 3–4 products churn the most.
- **Germany** has a noticeably higher churn rate than France and Spain.
- **Inactive members** churn much more often than active ones.
- **Age** is a strong driver: middle-aged customers (roughly 40–60) leave far more often than younger ones.

## Data leakage: the `Complain` column

Almost every customer who filed a complaint churned, and almost none of those who didn't. Including this column would give a near-perfect model, but it is misleading: a complaint is likely recorded during the exit process, so it wouldn't be available at prediction time. The column was removed from the features.

## Modeling

Features: all columns except IDs, surname, target and `Complain`. Categorical columns were one-hot encoded. Data split: 80% train / 20% test, stratified by the target.

| Model | Recall (churn) | ROC-AUC |
|---|---|---|
| Logistic Regression (scaled) | 0.21 | 0.778 |
| Random Forest (300 trees) | 0.47 | 0.867 |

Accuracy was not used as the main metric: a model that predicts "stays" for everyone would already reach ~80% accuracy while catching zero churners.

Random Forest performed much better because it captures non-linear patterns (like the number-of-products effect) that a linear model misses.

## Choosing the decision threshold

The default threshold of 0.5 is arbitrary. Lowering it catches more churners (higher recall) at the cost of more false alarms (lower precision). If a retention offer is cheap and losing a customer is expensive, a lower threshold (e.g. 0.3) is the better business choice. The right threshold depends on the cost of each type of error, not on a metric alone.

## Most important features

According to Random Forest: **Age**, **NumOfProducts** and **Balance**. Impurity-based importance tends to favour numeric features with many unique values, so binary features like `IsActiveMember` may be underrated; results were cross-checked with group-level churn rates.

## Business recommendation

Prioritise retention for customers aged 40–60, with 1 or 3–4 products, inactive, and from Germany. Use a low decision threshold, since contacting a customer who wasn't going to leave costs far less than losing one who was.

## Setup

1. Download `Customer-Churn-Records.csv` from the Kaggle link above and put it in the project folder
2. Install dependencies:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
3. Open `churn.ipynb` in VS Code or Jupyter and run the cells

## What I learned

- Always explore the data before modeling — the most important insight (data leakage) came from a simple group-by
- Accuracy is misleading on imbalanced data; recall, precision and ROC-AUC tell the real story
- Simple models are a baseline, not the final answer
- The decision threshold is a business decision, not a technical default
- Feature importance is a hint that needs to be verified with the data
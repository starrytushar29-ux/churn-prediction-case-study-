# Case Study 1: Customer Churn Prediction

**Dataset:** IBM Telco Customer Churn (7,043 customers, 21 features), included in `data/`.

**Approach:** Logistic Regression with L2 regularization.

**Pipeline:** load & inspect data -> handle missing TotalCharges -> check class balance ->
drop customerID -> encode target & one-hot encode categorical features -> stratified
train/test split -> StandardScaler (fit on train only) -> train Logistic Regression ->
evaluate (accuracy, precision, recall, F1, ROC-AUC, confusion matrix) -> sweep
regularization strength C in [0.01, 0.1, 1, 10, 100].

Run: open `Customer_Churn_Prediction.ipynb` and run all cells.

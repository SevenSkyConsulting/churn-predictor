from churn_predictor import xgboost, random_forest, logistic_regression

import pandas as pd

data = pd.read_csv('Telco_Customer_Churn.csv')
all_data = data
xgboost_accuracies = []
random_forest_accuracies = []
logistic_regression_accuracies = []

for i in range(10):
    all_data = all_data.sample(frac=1)
    train_data = all_data[100:]
    test_data = all_data[:100]
    X_train = train_data.drop(columns=['Churn', 'customerID', 'Contract', 'PaymentMethod'], axis=1)
    y_train = train_data['Churn']
    X_test = test_data.drop(columns=['Churn', 'customerID', 'Contract', 'PaymentMethod'], axis=1)
    y_test = test_data['Churn']

    xg = xgboost(X_train, y_train, X_test, y_test)
    rf = random_forest(X_train, y_train, X_test, y_test)
    lr = logistic_regression(X_train, y_train, X_test, y_test)

    xgboost_accuracies.append(xg)
    random_forest_accuracies.append(rf)
    logistic_regression_accuracies.append(lr)

print("XGBoost: ", sum(xgboost_accuracies)/len(xgboost_accuracies))
print("Random Forest: ", sum(random_forest_accuracies)/len(random_forest_accuracies))
print("Logistic Regression: ", sum(logistic_regression_accuracies)/len(logistic_regression_accuracies))

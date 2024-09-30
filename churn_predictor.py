from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def xgboost(train_x, train_y, test_x, test_y):
    xgboost_model = XGBClassifier()
    xgboost_model.fit(train_x, train_y)
    predict_y = xgboost_model.predict(test_x)
    accuracy_test = accuracy_score(test_y, predict_y)
    print('accuracy_score on test dataset : ', accuracy_test)
    return accuracy_test


def random_forest(train_x, train_y, test_x, test_y):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(train_x, train_y)
    y_predict = rf_classifier.predict(test_x)
    accuracy = accuracy_score(test_y, y_predict)
    print('accuracy_score on test dataset : ', accuracy)
    return accuracy


def logistic_regression(train_x, train_y, test_x, test_y):
    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=100)
    clf.fit(train_x, train_y)
    y_predict = clf.predict(test_x)
    accuracy = accuracy_score(test_y, y_predict)
    print('accuracy_score on test dataset : ', accuracy)
    return accuracy

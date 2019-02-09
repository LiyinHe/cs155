import numpy as np

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold

from google.colab import files
import io
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

data = files.upload()

# Read in data
data_train = pd.read_csv(io.BytesIO(data['train_2008.csv']))

# Extract input features and output labels
X = data_train.values[:, 1:-1]
y = data_train.values[:, -1]

# Scale data
X = preprocessing.scale(X)

# Specify number of folds for k-fold cross-validation
k = 5
folds = KFold(n_splits=k, shuffle=True, random_state=36)

# Test different regularization strengths for logistic regression
C_all = np.logspace(-2, 2, 9)
loss_all = np.empty(len(C_all))
for i, C in enumerate(C_all):
    loss = 0
    for train_inds, test_inds in folds.split(X):
        X_train, X_test = X[train_inds], X[test_inds]
        y_train, y_test = y[train_inds], y[test_inds]

        # Train logistic regression model
        logistic = LogisticRegression(C=C, solver='liblinear')
        logistic.fit(X_train, y_train)

        # Evaluate loss on validation set
        y_pred = logistic.predict_proba(X_test)
        loss += log_loss(y_test, y_pred)
    loss_all[i] = loss / k

# Extract best model
print(loss_all)
C = C_all[np.argmin(loss_all)]

# Specify number of folds for k-fold cross-validation
k = 3
folds = KFold(n_splits=k, shuffle=True, random_state=36)

loss = 0
for train_inds, test_inds in folds.split(X):
    X_train, X_test = X[train_inds], X[test_inds]
    y_train, y_test = y[train_inds], y[test_inds]

    # Train random forest model
    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(X_train, y_train)

    # Evaluate loss on validation set
    y_pred = forest.predict_proba(X_test)
    loss += log_loss(y_test, y_pred)
loss_all = loss / k

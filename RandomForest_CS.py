import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

import os
import pandas as pd
import pickle
import time


class RFmodel_CS:
    def __init__(self, rank=1, n_estimators=None):
        # Initialize model with hyperparameters of
        # specified performance rank (up to 10)
        with open('RandomForest_params.pkl', 'rb') as file:
            params, mean_test = pickle.load(file)
        model_params = params[rank - 1]
        if n_estimators is not None:
            model_params['n_estimators'] = n_estimators
        self.model = RandomForestClassifier(**model_params)

    def fit(self, X_train, y_train):
        self.model.fit(X_train[:, 1:], y_train)

    def predict(self, X_test):
        return self.model.predict_proba(X_test[:, 1:])[:, 1]

def main():
    # Read in training data
    data_train = pd.read_csv(os.path.join('data', 'train_2008.csv'))

    # Extract input features and output labels
    X_train = data_train.values[:, 1:-1]
    y_train = data_train.values[:, -1]

    # Define training set for hyperparameter selection
    inds = np.arange(len(X_train))
    X = X_train[inds]
    y = y_train[inds]

    # Specify hyperparameters for tuning
    parameters = {'n_estimators': np.arange(1000, 3100, 100),
                  'max_features': np.arange(15, 65, 5),
                  'min_samples_leaf': np.arange(0.0001, 0.005, 0.0001),
                  'max_depth': [None] + list(np.arange(10, 55, 5))}

    # Perform hyperparameter testing
    clf = RandomizedSearchCV(RandomForestClassifier(), parameters,
                             scoring='roc_auc', return_train_score=True,
                             n_iter=100, n_jobs=-1, pre_dispatch='2*n_jobs')
    clf.fit(X, y)

    # Save results for top 10 models
    inds_top = np.argsort(clf.cv_results_['rank_test_score'])[:10]
    params = np.array([clf.cv_results_['params'][i] for i in inds_top])
    mean_test = np.array([clf.cv_results_['mean_test_score'][i] for i in inds_top])
    model = clf.best_estimator_
    filename = 'RandomForest_{:s}.pkl'.format(time.strftime('%Y%m%d-%H%M'))
    with open(filename, 'wb') as file:
        pickle.dump((params, mean_test, model), file)
    with open('RandomForest_params.pkl', 'wb') as file:
        pickle.dump((params, mean_test), file)

    # Train best estimator
    if len(inds) < len(X_train):
        model.fit(X_train, y_train)

    # Show score on training set
    print('ROC AUC (training):',
          roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))

    # Read in input data for test sets
    test_2008 = pd.read_csv(os.path.join('data', 'test_2008.csv'))
    test_2012 = pd.read_csv(os.path.join('data', 'test_2012.csv'))

    # Make predictions on test sets
    pred_2008 = model.predict_proba(test_2008.values[:, 1:])[:, 1]
    pred_2012 = model.predict_proba(test_2012.values[:, 1:])[:, 1]

    # Write results
    df_2008 = pd.DataFrame(data={'id': test_2008.values[:, 0],
                                 'target': pred_2008})
    df_2008.to_csv(os.path.join('predictions', 'pred_2008_CS.csv'),
                   index=None, header=True)
    df_2012 = pd.DataFrame(data={'id': test_2012.values[:, 0],
                                 'target': pred_2012})
    df_2012.to_csv(os.path.join('predictions', 'pred_2012_CS.csv'),
                   index=None, header=True)

if __name__ == '__main__':
    main()

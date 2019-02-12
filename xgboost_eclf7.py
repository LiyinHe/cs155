import numpy as np
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

import pandas as pd


class xgbooster():
    def __init__(self, simpleFlag=True):
        if simpleFlag:
            n_estiList = [10] * 7
        else:
            n_estiList = [300, 310, 300, 315, 270, 255, 290]

        xg1 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.96, 
                                    learning_rate = 0.034, min_child_weight_s = 1.3300000000000003, 
                                    n_estimators = n_estiList[0], subsample = 0.905, 
                                    reg_lambda = 0.9900000000000001, min_split_loss = 0.01, max_depth = 12)

        xg2 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.95, 
                                    learning_rate = 0.049000000000000016, min_child_weight_s = 1.06, 
                                    n_estimators = n_estiList[1], subsample = 0.89, 
                                    reg_lambda = 1.0050000000000001, min_split_loss = 0.022, max_depth = 10)

        xg3 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.96, 
                                    learning_rate = 0.04499999999999999, min_child_weight_s = 1.3800000000000003, 
                                    n_estimators = n_estiList[2], subsample = 0.905, 
                                    reg_lambda = 1.06, min_split_loss = 0.015, max_depth = 10)

        xg4 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.99, 
                                    learning_rate = 0.039999999999999994, min_child_weight_s = 1.2300000000000004, 
                                    n_estimators = n_estiList[3], subsample = 0.91, 
                                    reg_lambda = 0.92, min_split_loss = 0.0, max_depth = 10)

        xg5 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.965, 
                                    learning_rate = 0.04999999999999999, min_child_weight_s = 1.3800000000000003, 
                                    n_estimators = n_estiList[4], subsample = 0.925, 
                                    reg_lambda = 1.04, min_split_loss = 0.015, max_depth = 11)

        xg6 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 1.03, 
                                    learning_rate = 0.04999999999999999, min_child_weight_s = 1.1400000000000001, 
                                    n_estimators = n_estiList[5], subsample = 0.915, 
                                    reg_lambda = 0.9800000000000001, min_split_loss = 0.0225, max_depth = 11)

        xg7 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.99, 
                                    learning_rate = 0.04200000000000001, min_child_weight_s = 1.06, 
                                    n_estimators = n_estiList[6], subsample = 0.9, 
                                    reg_lambda = 1.0050000000000001, min_split_loss = 0.0, max_depth = 11)

        if simpleFlag:
            estiList = [('xg1', xg1), ('xg2', xg2)]
        else:
            estiList = [('xg1', xg1), ('xg2', xg2),('xg3', xg3),('xg4', xg4), ('xg5', xg5),('xg6', xg6),('xg7', xg7)]

        
        eclf_7 = VotingClassifier(estimators=estiList, voting='soft', weights=[1]*len(estiList), flatten_transform=True)

        self.model = eclf_7

    def fit(self, X_train, Y_train):
        X_train_redu = X_train[:, 3:]
        self.model.fit(X_train_redu, Y_train)

    def predict(self, X_test):
        X_test_redu = X_test[:, 3:]
        return self.model.predict_proba(X_test_redu)[:, 1]
        

if __name__ == '__main__':
    df = pd.read_csv("./data/train_2008.csv")
    data = df.values

    X = data[:,:-1]
    Y = data[:,-1]

    df_test = pd.read_csv("./data/test_2008.csv")
    X_test = df_test.values

    eclf_7 = xgbooster(simpleFlag=True)
    eclf_7.fit(X, Y)

    y_predict = eclf_7.predict(X_test)

    submitPD = pd.DataFrame(y_predict, index=X_test[:, 0], columns=['target'])
    submitPD.to_csv('./predictions/xgPredicts_YM.csv', index_label='id')
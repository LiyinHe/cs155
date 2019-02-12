import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

import pandas as pd
df = pd.read_csv("train_2008.csv")
data = df.values

X = data[:,3:-1]
Y = data[:,-1]

xg1 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.96, 
                            learning_rate = 0.034, min_child_weight_s = 1.3300000000000003, 
                            n_estimators = 300, subsample = 0.905, 
                            reg_lambda = 0.9900000000000001, min_split_loss = 0.01, max_depth = 12)

xg2 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.95, 
                            learning_rate = 0.049000000000000016, min_child_weight_s = 1.06, 
                            n_estimators = 310, subsample = 0.89, 
                            reg_lambda = 1.0050000000000001, min_split_loss = 0.022, max_depth = 10)

xg3 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.96, 
                            learning_rate = 0.04499999999999999, min_child_weight_s = 1.3800000000000003, 
                            n_estimators = 300, subsample = 0.905, 
                            reg_lambda = 1.06, min_split_loss = 0.015, max_depth = 10)

xg4 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.99, 
                            learning_rate = 0.039999999999999994, min_child_weight_s = 1.2300000000000004, 
                            n_estimators = 315, subsample = 0.91, 
                            reg_lambda = 0.92, min_split_loss = 0.0, max_depth = 10)

xg5 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.965, 
                            learning_rate = 0.04999999999999999, min_child_weight_s = 1.3800000000000003, 
                            n_estimators = 270, subsample = 0.925, 
                            reg_lambda = 1.04, min_split_loss = 0.015, max_depth = 11)

xg6 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 1.03, 
                            learning_rate = 0.04999999999999999, min_child_weight_s = 1.1400000000000001, 
                            n_estimators = 255, subsample = 0.915, 
                            reg_lambda = 0.9800000000000001, min_split_loss = 0.0225, max_depth = 11)

xg7 = xgboost.XGBClassifier(eval_metric='auc', scale_pos_weight = 0.99, 
                            learning_rate = 0.04200000000000001, min_child_weight_s = 1.06, 
                            n_estimators = 290, subsample = 0.9, 
                            reg_lambda = 1.0050000000000001, min_split_loss = 0.0, max_depth = 11)


eclf_7 = VotingClassifier(estimators=[('xg1', xg1), ('xg2', xg2),('xg3', xg3),
                                                      ('xg4', xg4), ('xg5', xg5), 
                                                      ('xg6', xg6),('xg7', xg7)],voting='soft',weights=[1,1,1,1,1,1,1],
                                          flatten_transform=True)

import pandas as pd
df_test = pd.read_csv("test_2008.csv")
data_test = df_test.values

X_test = data_test[:,1:]

eclf_7.fit(X,Y)

y_predict = eclf_7.predict(X_test)

y_prob = eclf_7.predict_proba(X_test)

y_prob_1 = y_prob[:,1]

import pandas as pd
df_sub = pd.read_csv("sample_submission.csv")

data_sub = df_sub.values

for i in range(len(data_sub)):
    df_sub.loc[i,'target'] = y_prob_1[i]

export_csv = df_sub.to_csv ('export_dataframe_eclf7.csv', index = None, header=True) 
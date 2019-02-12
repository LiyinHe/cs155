import NeuroNetwork_YM
import RandomForest_CS
import xgboost_eclf7
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    # loading the training data
    df_train = np.random.permutation(np.array(pd.read_csv("./data/train_2008.csv")))
    X_train = df_train[:, 0:-1]
    Y_train = df_train[:, -1]

    # Collection of epoches 
    epIdice = [np.arange(a, a + 13000) for a in [0, 13000, 26000, 39000]]
    epIdice.append(np.arange(52000, len(X_train)))

    # Three classifiers j
    classifiers = [NeuroNetwork_YM.NNmodel_YM,
                   RandomForest_CS.RFmodel_CS,
                   xgboost_eclf7.xgbooster]

    pdctFeatures = np.zeros((len(X_train), len(classifiers)))

    for idx, epIdx in enumerate(epIdice):
        trainIdx = epIdice[0:idx] + epIdice[idx + 1:]
        trainIdx = np.hstack(trainIdx)

        xtrain = X_train[trainIdx]
        ytrain = Y_train[trainIdx]

        xcross = X_train[epIdx]

        for jdx, classifier in enumerate(classifiers):
            model = classifier()
            model.fit(xtrain, ytrain)
            pdctFeatures[epIdx, jdx] =  model.predict(xcross)

    ensLog = LogisticRegression()
    ensLog.fit(pdctFeatures, Y_train)

    df_test08 = np.array(pd.read_csv("./data/test_2008.csv"))
    X_test08 = df_test08

    df_test12 = np.array(pd.read_csv("./data/test_2012.csv"))
    X_test12 = df_test12

    feature08 = []
    feature12 = []
    trainedModel = []
    for classifier in classifiers:
        model = classifier()
        model.fit(X_train, Y_train)
        trainedModel.append(model)
        feature08.append(model.predict(X_test08))
        feature12.append(model.predict(X_test12))
    feature08 = np.vstack(feature08).T
    feature12 = np.vstack(feature12).T

    skPredicts08 = ensLog.predict_proba(feature08)[:, 1]
    skPredicts12 = ensLog.predict_proba(feature12)[:, 1]

    submitPD08 = pd.DataFrame(skPredicts08, index=X_test08[:, 0], columns=['target'])
    submitPD08.to_csv('./predictions/skPredicts08.csv', index_label='id')

    submitPD12 = pd.DataFrame(skPredicts12, index=X_test12[:, 0], columns=['target'])
    submitPD12.to_csv('./predictions/skPredicts12.csv', index_label='id')
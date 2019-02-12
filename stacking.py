import NeuroNetwork_YM
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    df_train = np.random.permutation(np.array(pd.read_csv("./data/train_2008.csv")))
    X_train = df_train[:, 0:-1]
    Y_train = df_train[:, -1]

    epIdice = [np.arange(a, a + 13000) for a in [0, 13000, 26000, 39000]]
    epIdice.append(np.arange(52000, len(X_train)))

    classifiers = [NeuroNetwork_YM.NNmodel_YM, NeuroNetwork_YM.NNmodel_YM]
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

    feature08 = []
    trainedModel = []
    for classifier in classifiers:
        model = classifier()
        model.fit(X_train, Y_train)
        trainedModel.append(model)
        feature08.append(model.predict(X_test08))
    feature08 = np.hstack(feature08)

    skPredicts = ensLog.predict_proba(feature08)[:, 1]

    submitPD = pd.DataFrame(skPredicts, index=X_test08[:, 0], columns=['target'])
    submitPD.to_csv('./predictions/skPredicts_YM.csv', index_label='id')


    # eclf = VotingClassifier(estimators=[('NN1', Sequential()), ('NN2', Sequential())])
    # eclf.fit(X_train, Y_train)


    # print(epIdx)
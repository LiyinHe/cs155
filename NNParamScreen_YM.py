import NeuroNetwork_YM
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df_train = np.random.permutation(np.array(pd.read_csv("./data/train_2008.csv")))
    X_train = df_train[0:55000, 0:-1]
    Y_train = df_train[0:55000, -1]

    X_cross = df_train[55000:, 0:-1]
    Y_cross = df_train[55000:, -1]

    df_test08 = np.array(pd.read_csv("./data/test_2008.csv"))
    X_test08 = df_test08[:, 0:-1]

    trainAcc = []
    crossAcc = []
    for dropRate in np.linspace(0, 0.99, 20):
        print("rate at {}".format(dropRate))
        nnmodel = NeuroNetwork_YM.NNmodel_YM(dpRate=dropRate)
        nnmodel.fit(X_train, Y_train, epoch=200, verb=1)
        trainAcc.append(nnmodel.evaluate(X_train, Y_train)[1])
        crossAcc.append(nnmodel.evaluate(X_cross, Y_cross)[1])

    # nnPredicts = nnmodel.predict(X_test08)

    # submitPD = pd.DataFrame(nnPredicts, index=X_test08[:, 0], columns=['target'])
    # submitPD.to_csv('./preDicts/nnPredicts_YM.csv', index_label='id')


    # eclf = VotingClassifier(estimators=[('NN1', Sequential()), ('NN2', Sequential())])
    # eclf.fit(X_train, Y_train)

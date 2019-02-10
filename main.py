import NeuroNetwork_YM
import pandas as pd
import numpy as np

from keras.models import Sequential

from sklearn.ensemble import VotingClassifier

if __name__ == "__main__":
    df_train = np.array(pd.read_csv("./data/train_2008.csv"))
    X_train = df_train[:, 0:-1]
    Y_train = df_train[:, -1]

    df_test08 = np.array(pd.read_csv("./data/test_2008.csv"))
    X_test08 = df_test08[:, 0:-1]

    nnmodel = NeuroNetwork_YM.NNmodel_YM()
    nnmodel.fit(X_train, Y_train, epoch=50, verb=1)

    nnPredicts = nnmodel.predict(X_test08)

    submitPD = pd.DataFrame(nnPredicts, index=X_test08[:, 0], columns=['target'])
    submitPD.to_csv('./preDicts/nnPredicts_YM.csv', index_label='id')


    # eclf = VotingClassifier(estimators=[('NN1', Sequential()), ('NN2', Sequential())])
    # eclf.fit(X_train, Y_train)

import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras import regularizers

# Entropy calculation
def ent(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

# return a list of filtered features
def generate_idx50():
    df_train = np.array(pd.read_csv("./data/train_2008.csv"))
    X_train = df_train[:, 0:-1]
    Y_train = df_train[:, -1]
    entros = []
    MIs = []
    for feature in X_train.T:
        entros.append(ent(feature, 2))
        MIs.append(mutual_info_score(feature, Y_train))

    flags1 = entros > np.percentile(entros, 50) 
    flags2 = MIs > np.percentile(MIs, 50) 
    flags = [a[0] or a[1] for a in zip(flags1, flags2)]
    flags[0] = False
    flags[23] = False
    idx50 = np.where(flags)[0]

    return idx50

# Reduce samample X with 
def reduceX(X):
    X_reduced = [a for idx, a in enumerate(X.T) if idx in idx50]
    X_reduced = np.array(X_reduced).T
    print(X_reduced.shape)
    return X_reduced

idx50 = generate_idx50()

class NNmodel_YM:
    def __init__(self, dpRate=0.5):
        self.normFactor = 1
        
        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(idx50),)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(dpRate))
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dense(32))
        self.model.add(Dropout(dpRate))
        self.model.add(Activation('relu'))
        self.model.add(Dense(16))
        self.model.add(Activation('relu'))
        self.model.add(Dense(4))
        self.model.add(Dropout(dpRate))
        self.model.add(Activation('relu'))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
        # self.model.save('./NNtemp/UntrainedModel.h5')

    def fit(self, X_train, Y_train, epoch=50, verb=1):
        X_train_reduced = reduceX(X_train)
        self.normFactor = np.std(X_train_reduced, 0)
        X_train_reduced = X_train_reduced / self.normFactor
        fit = self.model.fit(X_train_reduced, Y_train, batch_size=128, nb_epoch=epoch, verbose=verb)
        # self.model.save('./NNtemp/TrainedModel.h5')

    def predict(self, X_test):
        X_test_reduced = reduceX(X_test) / self.normFactor
        return self.model.predict(X_test_reduced)

    def evaluate(self, X_test, Y_test):
        X_test_reduced = reduceX(X_test) / self.normFactor
        return self.model.evaluate(X_test_reduced, Y_test)



if __name__ == '__main__':
    df_train = np.array(pd.read_csv("./data/train_2008.csv"))
    X_train = df_train[:, 0:-1]
    Y_train = df_train[:, -1]

    nnmodel = NNmodel_YM()
    nnmodel.fit(X_train, Y_train, epoch=20, verb=1)

    trainAcc = nnmodel.evaluate(X_train, Y_train)[1]
    print('Training Acc: {0:.3f}'.format(trainAcc))

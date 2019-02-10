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

def ent(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

df_train = np.array(pd.read_csv("./data/train_2008.csv"))
X_train = df_train[:, 0:-1]
Y_train = df_train[:, -1]

df_test = np.array(pd.read_csv("./data/test_2008.csv"))
X_test = df_test[:, 0:-1]
Y_test = df_test[:, -1]

entros = []
MIs = []
for feature in X_train.T:
    entros.append(ent(feature, 2))
    MIs.append(mutual_info_score(feature, Y_train))

flags1 = entros > np.percentile(entros, 50) 
flags2 = MIs > np.percentile(MIs, 50) 
flags = [a[0] & a[1] for a in zip(flags1, flags2)]
flags[0] = False
flags[23] = False
idx50 = np.where(flags)[0]

X_train_reduced = [a for idx, a in enumerate(X_train.T) if idx in idx50]
X_train_reduced = np.array(X_train_reduced).T

X_test_reduced = [a for idx, a in enumerate(X_test.T) if idx in idx50]
X_test_reduced = np.array(X_test_reduced).T

normFactor = np.std(X_train_reduced, 0)
X_train_reduced = X_train_reduced / normFactor
X_test_reduced = X_test_reduced / normFactor


# Neuron Network 
model = Sequential()

model.add(Dense(128, input_shape=(X_train_reduced.shape[1],)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(optimizer='adam', loss='mean_squared_logarithmic_error', metrics=['accuracy'])
print("Parameter number at {0}".format(model.count_params()))

fit = model.fit(X_train_reduced, Y_train, batch_size=128, nb_epoch=1000, verbose=0)

# print(model.predict(X_train_reduced[0:20]))
trainAcc, testAcc = (model.evaluate(x=X_train_reduced, y=Y_train)[1], model.evaluate(x=X_test_reduced, y=Y_test)[1])
print('Training Acc: {0:.3f}, Testing Acc: {1:.3f}'.format(testAcc, trainAcc))
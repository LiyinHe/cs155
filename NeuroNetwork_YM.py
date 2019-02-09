import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

def ent(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

df_test = np.array(pd.read_csv("./data/train_2008.csv"))

X_test = df_test[:, 0:-1]
Y_test = df_test[:, -1]

entros = []
MIs = []
for feature in X_test.T:
    entros.append(ent(feature, 2))
    MIs.append(mutual_info_score(feature, Y_test))

flags1 = entros > np.percentile(entros, 50) 
flags2 = MIs > np.percentile(MIs, 50) 
flags = [a[0] & a[1] for a in zip(flags1, flags2)]
flags[0] = False
flags[23] = False
idx50 = np.where(flags)[0]

X_test_reduced = [a for idx, a in enumerate(X_test.T) if idx in idx50]
X_test_reduced = np.array(X_test_reduced)

print(X_test_reduced.shape)
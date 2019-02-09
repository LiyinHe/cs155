import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

def ent(labels, base=None):
  value, counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

df_test = np.array(pd.read_csv("./data/train_2008.csv"))

X_test = df_test[:, 0:-1]
Y_test = df_test[-1]


entros = []
for feature in X_test:
    entros.append(ent(feature))

plt.bar(list(range(len(entros))), entros)
plt.xlabel('Feature Index')
plt.ylabel('entropy')

plt.show()
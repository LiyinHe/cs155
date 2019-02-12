import numpy as np
import pandas as pd
import os


# Read in data
pred_LY = pd.read_csv('export_dataframe.csv')
pred_YM = pd.read_csv(os.path.join('predictions', 'nnPredicts_YM.csv'))
pred_CS = pd.read_csv(os.path.join('predictions', 'pred_2008_CS.csv'))

# Average predictions
a, b, c = np.ones(3) / 3
pred = a * pred_LY.values[:, 1] + b * pred_YM.values[:, 1] + c * pred_CS.values[:, 1]

# Write out results
df_2008 = pd.DataFrame(data={'id': pred_CS.values[:, 0].astype(int),
                             'target': pred})
df_2008.to_csv(os.path.join('predictions', 'pred_2008_avg.csv'),
               index=None, header=True)
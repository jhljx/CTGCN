# coding: utf-8
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

time_list = ['2004-04', '2004-05', '2004-06', '2004-07', '2004-08', '2004-09', '2004-10']
method = 'EvolveGCN'
for time in time_list:
    df = pd.read_csv(time + '.csv', sep='\t', index_col=0)
    df_core_num = pd.read_csv('../' + time + '_core_num.csv')
    x = df.loc[df_core_num.node.tolist(), :].values
    x_reduce = tsne.fit_transform(x)
    np.savetxt(time + '-' + method + '.txt', x_reduce)
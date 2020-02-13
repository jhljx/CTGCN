import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import os
from sklearn import metrics

def visualization(df, true_label, tsne_method_path, vis_method_path, date, reduction='tsne'):
    if reduction == 'tsne':
        method = TSNE(n_components=2)
        reduce_arr = method.fit_transform(df)  # 进行数据降维
    elif reduction == 'pca':
        method = PCA(n_components=2)
        reduce_arr = method.fit_transform(df)  # 进行数据降维
    else:
        raise Exception('reduction parameter error！')

    df_reduce = pd.DataFrame(reduce_arr, index=df.index, columns=['comp1', 'comp2'])
    df_reduce['label'] = true_label.apply(lambda x: 'B' if x == 0 else 'A')
    df_reduce.to_csv(os.path.join(tsne_method_path, date + '_' + reduction + '.csv'), sep=',', index=False)
    reduce_mat = df_reduce.values

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    color_list = true_label.map({1: 'r', 2: 'r', 0:'b'}).tolist()
    ax.scatter(reduce_mat[:, 0], reduce_mat[:, 1], c=color_list)
    plt.savefig(os.path.join(vis_method_path, date + '.png'), dpi=300)
    plt.close('all')

if __name__ == '__main__':

    reduction_method = 'tsne'

    label_path = 'data\\email-eu\\label'
    tsne_result_path = 'data\\email-eu\\' + reduction_method
    visualization_path = 'data\\email-eu\\' + reduction_method + '_visualization'
    embedding_path = 'data\\email-eu\\2.embedding'

    for method in os.listdir(embedding_path):
        print('Current method is :{}'.format(method))
        method_path = os.path.join(embedding_path, method)
        files = os.listdir(method_path)
        tsne_method_path = os.path.join(tsne_result_path, method)
        if not os.path.exists(tsne_method_path):
            os.makedirs(tsne_method_path)
        vis_method_path = os.path.join(visualization_path, method)
        if not os.path.exists(vis_method_path):
            os.makedirs(vis_method_path)
        date_dict = {}
        for file in files:
            date = os.path.splitext(file)[0]
            print('Current date is :{}'.format(date))

            data_df = pd.read_csv(os.path.join(method_path, file), sep='\t', index_col=0)
            label_df = pd.read_csv(os.path.join(label_path, file), sep='\t', index_col=0)
            data_merge = pd.merge(data_df, label_df, how='inner', left_index=True, right_index=True).reset_index(drop=True)
            data_merge.dropna()
            label_df = data_merge['label']
            data_df = data_merge.drop(['label'], axis=1)
            assert label_df.shape[0] == data_df.shape[0]
            visualization(data_df, label_df, tsne_method_path, vis_method_path, date, reduction=reduction_method)

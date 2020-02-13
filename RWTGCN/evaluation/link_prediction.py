import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from numpy import random

class DataGenerator(object):
    base_path: str
    input_base_path: str
    output_base_path: str
    def __init__(self, base_path, input_folder, output_folder, ):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)
        return

    def generate_edge_samples(self, input_dir, output_dir, node_dir_path):
        print('Start geerate edge samples!')
        node_set_path = os.path.join(node_dir_path, 'nodes.csv')
        df_nodes = pd.read_csv(node_set_path, sep=',', header=None, names=['node'])
        # print(df_nodes.head())
        node_num = df_nodes.shape[0]

        file_name_list = os.listdir(input_dir)
        for file_name in file_name_list:
            file_path = os.path.join(input_dir, file_name)
            df_edge = pd.read_csv(file_path, sep='\t')
            graph_dict = dict()

            # 用dict模拟图数据结构
            def func(edge_series, graph=None):
                from_id = edge_series['from_id']
                to_id = edge_series['to_id']
                if from_id not in graph_dict:
                    graph_dict[from_id] = dict()
                    graph_dict[from_id][to_id] = 1
                else:
                    graph_dict[from_id][to_id] = 1
                if to_id not in graph_dict:
                    graph_dict[to_id] = dict()
                    graph_dict[to_id][from_id] = 1
                else:
                    graph_dict[to_id][from_id] = 1

            df_edge.apply(func, graph=graph_dict, axis=1)

            output_list = []
            # 模拟产生正负样本
            for from_id, neighbor_dict in graph_dict.items():
                for to_id, val in neighbor_dict.items():
                    output_list.append([from_id, to_id, 1])
                    for i in range(10):
                        node_idx = random.choice(node_num, 1)[0]
                        node_id = df_nodes.at[node_idx, 'node']
                        if node_id not in neighbor_dict:
                            output_list.append([from_id, node_id, 0])
                            break
            output_path = os.path.join(output_dir, file_name)
            df_output = pd.DataFrame(output_list, columns=['from_id', 'to_id', 'label'])
            df_output.to_csv(output_path, sep=',', index=False)
        print('Generate edge samples finish!')

    def generate_edge_features(self, embedding_dir_path, edge_dir_path, output_dir):
        print('Start generate edge features!')
        method_list = os.listdir(embedding_dir_path)
        # method_run_list = ['deepwalk', 'line', 'node2vec', 'struct2vec', 'dyGEM', 'dyTriad', 'timers']
        for method in method_list:
            # if method in ['dyGEM_model_weight', 'dyTriad_format']:
            #    continue
            # if method != 'dynspe_embedding_alpha_10000--beta_0.0001':
            #    continue
            print('Current method is :{}'.format(method))
            method_path = os.path.join(embedding_dir_path, method)
            edge_file_list = os.listdir(edge_dir_path)
            edge_file_num = len(edge_file_list)

            # 按不同时间片
            for i in range(edge_file_num):
                file = edge_file_list[i]
                print('Current date is :{}'.format(file))
                # 当前方法的嵌入结果
                # 默认把第一行作为header
                if method == 'dyTriad':
                    df_embedding = pd.read_csv(os.path.join(method_path, str(i) + '.out'), sep=' ', index_col=0,
                                               header=None)
                else:
                    df_embedding = pd.read_csv(os.path.join(method_path, file), sep='\t', index_col=0)
                # print(df_embedding.head())
                # exit(0)
                print('read edge sample!')
                df_edge = pd.read_csv(os.path.join(edge_dir_path, file))
                # 获取edge的feature矩阵数据
                date = file.split('.')[0]
                output_method_dir = os.path.join(output_dir, method, date)
                if not os.path.exists(output_method_dir):
                    os.makedirs(output_method_dir)
                self.generate_edge_feature_matrix(df_embedding, df_edge, output_method_dir)
        print('Finish generate edge features!')

    def generate_edge_feature_matrix(self, df_embedding, df_edge, output_dir):
        print('Start generate edge feature matrix!')

        def average_func(node_seris, data=None):
            from_id = node_seris['from_id']
            to_id = node_seris['to_id']
            from_vec, to_vec = data.loc[from_id], data.loc[to_id]
            return (from_vec + to_vec) / 2

        def hadmard_func(node_seris, data=None):
            from_id = node_seris['from_id']
            to_id = node_seris['to_id']
            from_vec, to_vec = data.loc[from_id], data.loc[to_id]
            return from_vec * to_vec

        def weight_l1(node_seris, data=None):
            from_id = node_seris['from_id']
            to_id = node_seris['to_id']
            from_vec, to_vec = data.loc[from_id], data.loc[to_id]
            return (from_vec - to_vec).abs()

        def weight_l2(node_seris, data=None):
            from_id = node_seris['from_id']
            to_id = node_seris['to_id']
            from_vec, to_vec = data.loc[from_id], data.loc[to_id]
            return (from_vec - to_vec).pow(2)

        average_data = df_edge.apply(average_func, data=df_embedding, axis=1)
        hadmard_data = df_edge.apply(hadmard_func, data=df_embedding, axis=1)
        weight_l1_data = df_edge.apply(weight_l1, data=df_embedding, axis=1)
        weight_l2_data = df_edge.apply(weight_l2, data=df_embedding, axis=1)

        average_df = pd.DataFrame(average_data)
        hadmard_df = pd.DataFrame(hadmard_data)
        weight_l1_df = pd.DataFrame(weight_l1_data)
        weight_l2_df = pd.DataFrame(weight_l2_data)

        average_df.to_csv(os.path.join(output_dir, 'average_edge_feature.csv'), sep=',', index=False)
        hadmard_df.to_csv(os.path.join(output_dir, 'hadmard_edge_feature.csv'), sep=',', index=False)
        weight_l1_df.to_csv(os.path.join(output_dir, 'weight_l1_edge_feature.csv'), sep=',', index=False)
        weight_l2_df.to_csv(os.path.join(output_dir, 'weight_l2_edge_feature.csv'), sep=',', index=False)
        print('Finish generate edge feature matrix!')

class LinkPredictor(object):
    base_path: str
    input_base_path: str
    output_base_path: str

    def __init__(self, base_path, input_folder, output_folder, ):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)
        return

    def train(self, data_path, train_label, model_dict):
        print('Start training!')
        average_df = pd.read_csv(os.path.join(data_path, 'average_edge_feature.csv'))
        hadmard_df = pd.read_csv(os.path.join(data_path, 'hadmard_edge_feature.csv'))
        weight_l1_df = pd.read_csv(os.path.join(data_path, 'weight_l1_edge_feature.csv'))
        weight_l2_df = pd.read_csv(os.path.join(data_path, 'weight_l2_edge_feature.csv'))

        average_model = LogisticRegression()
        average_model.fit(average_df, train_label)
        model_dict['average'] = average_model

        hadmard_model = LogisticRegression()
        hadmard_model.fit(hadmard_df, train_label)
        model_dict['hadmard'] = hadmard_model

        weight_l1_model = LogisticRegression()
        weight_l1_model.fit(weight_l1_df, train_label)
        model_dict['weight_l1'] = weight_l1_model

        weight_l2_model = LogisticRegression()
        weight_l2_model.fit(weight_l2_df, train_label)
        model_dict['weight_l2'] = weight_l2_model
        print('Finish training!')

    def test(self, data_path, test_label, model_dict, date_dict, date):
        print('Start testing!')
        average_df = pd.read_csv(os.path.join(data_path, 'average_edge_feature.csv'))
        hadmard_df = pd.read_csv(os.path.join(data_path, 'hadmard_edge_feature.csv'))
        weight_l1_df = pd.read_csv(os.path.join(data_path, 'weight_l1_edge_feature.csv'))
        weight_l2_df = pd.read_csv(os.path.join(data_path, 'weight_l2_edge_feature.csv'))

        average_pred = model_dict['average'].predict_proba(average_df)[:, 1]
        hadmard_pred = model_dict['hadmard'].predict_proba(hadmard_df)[:, 1]
        weight_l1_pred = model_dict['weight_l1'].predict_proba(weight_l1_df)[:, 1]
        weight_l2_pred = model_dict['weight_l2'].predict_proba(weight_l2_df)[:, 1]

        auc_dict = dict()
        average_auc_score = roc_auc_score(test_label, average_pred)
        auc_dict['average'] = average_auc_score
        hadmard_auc_score = roc_auc_score(test_label, hadmard_pred)
        auc_dict['hadmard'] = hadmard_auc_score
        weight_l1_auc_score = roc_auc_score(test_label, weight_l1_pred)
        auc_dict['weight_l1'] = weight_l1_auc_score
        weight_l2_auc_score = roc_auc_score(test_label, weight_l2_pred)
        auc_dict['weight_l2'] = weight_l2_auc_score

        date_dict[date] = auc_dict
        print('Finish testing!')

    def link_prediction(self, embedding_dir_path, edge_dir_path, edge_feature_dir, link_prediction_path):
        print('Start link prediction!')
        method_list = os.listdir(embedding_dir_path)
        #method_run_list = ['deepwalk', 'line', 'node2vec', 'struct2vec', 'dyGEM', 'dyTriad', 'timers']
        for method in method_list:
            # if method in ['dyGEM_model_weight', 'dyTriad_format']:
            #    continue
            #if method != 'dynspe_embedding_alpha_10000--beta_0.0001':
            #    continue
            print('Current method is :{}'.format(method))
            # if method not in ['dynspe']:
            #     continue
            #method_path = os.path.join(embedding_dir_path, method)

            edge_file_list = os.listdir(edge_dir_path)
            # lp_method_path = os.path.join(link_prediction_path, method)
            # if not os.path.exists(lp_method_path):
            #      os.makedirs(lp_method_path)

            edge_file_num = len(edge_file_list)
            model_dict = dict()
            auc_dict = dict()

            # 按不同时间片
            for i in range(edge_file_num):
                file = edge_file_list[i]
                print('Current date is :{}'.format(file))
                date = file.split('.')[0]
                # lp_method_date_path = os.path.join(os.path.join(link_prediction_path, method), date)
                # if not os.path.exists(lp_method_date_path):
                #     os.makedirs(lp_method_date_path)
                # 当前方法的嵌入结果
                # 默认把第一行作为header
                df_edge = pd.read_csv(os.path.join(edge_dir_path, file))
                edge_feature_path = os.path.join(edge_feature_dir, method, date)
                train_label = df_edge['label']

                if i != 0:
                    # 第一个时间片不用测试数据
                    #print('Start testing')
                    self.test(edge_feature_path, train_label, model_dict, auc_dict, date)

                if i != edge_file_num - 1:
                    # 最后一个时间片不用训练数据
                    #print('Start training')
                    self.train(edge_feature_path, train_label, model_dict)
            print(auc_dict)
            pd.DataFrame(auc_dict).T.to_csv(os.path.join(link_prediction_path, method + '_auc_record.csv'), sep=',')
        print('Finish link prediction!')

if __name__ == '__main__':

    data_set_name = 'enron_test'
    data_generator = DataGenerator()

    anomaly_detector = LinkPredictor()
    #print(os.getcwd())
    # root_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    # data_path = os.path.join(root_path, 'data')
    #
    # node_dir_path = os.path.join(data_path, data_set_name + '/nodes_set')
    # edge_dir_path = os.path.join(data_path, data_set_name + '/1.format')
    # embedding_path = os.path.join(data_path, data_set_name +  '/2.embedding')
    # edge_sample_dir_path = os.path.join(data_path, data_set_name + '/edge_sample')
    # edge_feature_dir_path = os.path.join(data_path, data_set_name + '/edge_feature')
    #
    # link_prediction_path =os.path.join(data_path,'link_pred_' + data_set_name)
    #
    # if not os.path.exists(link_prediction_path):
    #     os.makedirs(link_prediction_path)
    # if not os.path.exists(edge_sample_dir_path):
    #     os.makedirs(edge_sample_dir_path)
    #
    # # 1. 先通过采样获取正负样本边集
    # #generate_edge_samples(edge_dir_path, edge_sample_dir_path, node_dir_path)
    #
    #
    # # 2. 再遍历每种方法得到相应时刻的边集对应的embedding特征
    # generate_edge_features(embedding_path, edge_sample_dir_path, edge_feature_dir_path)
    #
    # # 3. 对每个embedding方法，训练并测试其每个时间片的link LR模型
    # link_prediction(embedding_path, edge_sample_dir_path, edge_feature_dir_path, link_prediction_path)



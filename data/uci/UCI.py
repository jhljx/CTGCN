#coding: utf-8
import numpy as np
import pandas as pd
import os
from datetime import datetime


def generate_format_data(input_file_path, output_base_path, output_graph_dir, output_node_dir):
    output_graph_dir_path = os.path.join(output_base_path, output_graph_dir)
    output_node_dir_path = os.path.join(output_base_path, output_node_dir)

    if not os.path.exists(output_graph_dir_path):
        os.makedirs(output_graph_dir_path)
    if not os.path.exists(output_node_dir_path):
        os.makedirs(output_node_dir_path)

    node_dict = {}
    df = pd.read_csv(input_file_path, sep=' ', header=None, skiprows=2, names=['from_id', 'to_id', 'weight', 'time'])
    df['time'] = df['time'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m'))

    for time, df_time in df.groupby(by=['time']):
        print('time = ', time)
        df_output = df_time.drop(['time'], axis=1)

        def trans_id(nid):
            node_dict['U' + str(nid)] = 1
            return 'U' + str(nid)

        df_output[['from_id', 'to_id']] = df_output[['from_id', 'to_id']].applymap(trans_id)
        datestamp = str(time)
        output_file_path = os.path.join(output_graph_dir_path, datestamp + '.csv')
        df_output.to_csv(output_file_path, sep='\t', index=False)

    node_list = list(node_dict.keys())
    node_list = sorted(node_list)
    df_node = pd.DataFrame(node_list, columns=['node'])
    node_file_path = os.path.join(output_node_dir_path, 'nodes.csv')
    df_node.to_csv(node_file_path, sep='\t', index=False, header=False)


if __name__ == '__main__':
    input_file_path = '/data/uci/0.input/graph.txt'
    output_base_path = '/data/uci'
    output_graph_dir = '1.format'
    output_node_dir = 'nodes_set'
    generate_format_data(input_file_path, output_base_path, output_graph_dir, output_node_dir)
import numpy as np
import pandas as pd
import os

def build_dynamic_graph(file_path, output_dir, node_dir, sep='\t', graph_num=10):
    df_graph = pd.read_csv(file_path, sep=sep, header=None, dtype=str)
    tot_num = df_graph.shape[0]
    col_num = df_graph.shape[1]
    if col_num == 2:
        df_graph.columns = ['from_id', 'to_id']
        df_graph['weight'] = 1
    elif col_num == 3:
        df_graph.columns = ['from_id', 'to_id', 'weight']
        df_graph['weight'] = df_graph['weight'].apply(np.float)
    else:
        raise Exception('Unsupported input file format.')
    idx_arr = np.random.permutation(np.arange(tot_num))
    df_graph = df_graph.loc[idx_arr, :].reset_index(drop=True)
    df_graph['from_id'] = df_graph['from_id'].apply(lambda x: 'U' + x)
    df_graph['to_id'] = df_graph['to_id'].apply(lambda x: 'U' + x)

    node_arr = pd.concat([df_graph['from_id'], df_graph['to_id']], axis=0).unique()
    node_arr.sort()
    df_node = pd.DataFrame(node_arr, columns=['node'])
    df_node.to_csv(os.path.join(node_dir, 'nodes.csv'), sep='\t', index=False, header=False)

    base_num = tot_num // graph_num
    if tot_num % graph_num == 0:
        pos = base_num - 1
    else:
        pos = base_num + tot_num % graph_num - 1
    df_graph.loc[:pos, :].to_csv(os.path.join(output_dir, '0.csv'), sep='\t', index=False)
    for i in range(1, graph_num):
        df_graph.loc[:pos + base_num * i, :].to_csv(os.path.join(output_dir, str(i) + '.csv'), sep='\t', index=False)
    return


def transform():
    input_dir = '/data/america_air/0.input'
    output_dir = '/data/america_air/1.format'
    node_dir = '/data/america_air/nodes_set'
    intput_path = os.path.join(input_dir, 'usa-airports.edgelist')
    build_dynamic_graph(intput_path, output_dir, node_dir, sep=' ')

    input_dir = '/data/europe_air/0.input'
    output_dir = '/data/europe_air/1.format'
    node_dir = '/data/europe_air/nodes_set'
    intput_path = os.path.join(input_dir, 'europe-airports.edgelist')
    build_dynamic_graph(intput_path, output_dir, node_dir, sep=' ')

    input_dir = '/data/jazz/0.input'
    output_dir = '/data/jazz/1.format'
    node_dir = '/data/jazz/nodes_set'
    intput_path = os.path.join(input_dir, 'jazz.csv')
    build_dynamic_graph(intput_path, output_dir, node_dir, sep='\t')

    input_dir = '/data/blogcatalog/0.input'
    output_dir = '/data/blogcatalog/1.format'
    node_dir = '/data/blogcatalog/nodes_set'
    intput_path = os.path.join(input_dir, 'blogcatalog.csv')
    build_dynamic_graph(intput_path, output_dir, node_dir, sep=',')

if __name__ == '__main__':
    transform()
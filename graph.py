# coding: utf-8
import numpy as np
import pandas as pd
import os
import networkx as nx
from utils import get_nx_graph, check_and_make_path


def get_graph_from_nodes(file_path, node_file, output_node_dir, output_edge_dir, sep='\t'):
    import random
    df_edges = pd.read_csv(file_path, sep=sep, header=0)
    # node_list = pd.unique(pd.concat([df_edges['from_id'], df_edges['to_id']], axis=0)).tolist()
    nodes_set = pd.read_csv(node_file, names=['node'])
    full_node_list = nodes_set['node'].tolist()
    print('node number: ', len(full_node_list))
    check_and_make_path(output_node_dir)
    check_and_make_path(output_edge_dir)
    nx_graph = get_nx_graph(file_path, full_node_list, sep=sep)
    node_num_list = [50, 100, 500, 1000, 5000, 10000]
    max_cc = max(nx.connected_components(nx_graph), key=len)
    node_list = list(max_cc)
    print(node_list[:10])
    print(len(node_list))
    for i, node_num in enumerate(node_num_list):
        start_node = random.sample(node_list, 1)[0]
        adj = nx_graph.adj
        node_dict = dict()
        node_dict[start_node] = 1
        sample_list = [start_node]
        front, cnt = -1, 1
        while front < cnt and cnt < node_num:
            front += 1
            # print('front = ', front)
            cur = sample_list[front]
            for neighbor, edge_attr in adj[cur].items():
                if neighbor not in node_dict:
                    node_dict[neighbor] = 1
                    cnt += 1
                    sample_list.append(neighbor)
                    if cnt >= node_num:
                        break
            if cnt > node_num:
                break
        # print(sample_nodes)
        print('i = ', i, 'cnt = ', cnt)
        nx_subgraph = nx_graph.subgraph(sample_list)
        edge_list = []
        df_nodes = pd.DataFrame(sample_list, columns=['node'])
        df_nodes.to_csv(os.path.join(output_node_dir, str(i) + '.csv'), sep='\t', index=False, header=False)
        for node, neighbors in nx_subgraph.adj.items():
            for neighbor, edge_attr in neighbors.items():
                edge_list.append([node, neighbor, edge_attr['weight']])
        edges_arr = np.array(edge_list)
        print('edges arr shape: ', edges_arr.shape[0])
        df_output = pd.DataFrame(edges_arr, columns=['from_id', 'to_id', 'weight'])
        df_output.to_csv(os.path.join(output_edge_dir, str(i) + '.csv'), sep='\t', index=False)
    df_nodes = pd.DataFrame(np.array(full_node_list), columns=['node'])
    df_nodes.to_csv(os.path.join(output_node_dir, str(len(node_num_list)) + '.csv'), sep='\t', index=False, header=False)
    df_edges.to_csv(os.path.join(output_edge_dir, str(len(node_num_list)) + '.csv'), sep='\t', index=False)


def get_graph_from_edges(file_path, node_file, output_node_dir, output_edge_dir, sep='\t'):
    import random
    df_edges = pd.read_csv(file_path, sep=sep, header=0)
    all_edge_num = df_edges.shape[0]
    check_and_make_path(output_node_dir)
    check_and_make_path(output_edge_dir)
    edge_num_list = [50, 100, 500, 1000, 5000, 10000, 70000]
    edge_indices = np.arange(all_edge_num).tolist()
    for i, edge_num in enumerate(edge_num_list):
        sample_edge_indices = random.sample(edge_indices, edge_num)
        df_subgraph = df_edges.loc[sample_edge_indices, :]
        node_list = pd.unique(pd.concat([df_subgraph['from_id'], df_subgraph['to_id']], axis=0)).tolist()
        df_nodes = pd.DataFrame(node_list, columns=['node'])
        df_nodes.to_csv(os.path.join(output_node_dir, str(i) + '.csv'), sep='\t', index=False)
        df_subgraph.to_csv(os.path.join(output_edge_dir, str(i) + '.csv'), sep='\t', index=False)


def build_dynamic_graph(file_path, output_dir, node_dir, sep='\t', graph_num=10):
    df_graph = pd.read_csv(file_path, sep=sep, header=None, dtype=str)
    tot_num = df_graph.shape[0]
    col_num = df_graph.shape[1]
    assert col_num in [2, 3]
    if col_num == 2:
        df_graph.columns = ['from_id', 'to_id']
        df_graph['weight'] = 1
    else:  # col_num == 3
        df_graph.columns = ['from_id', 'to_id', 'weight']
        df_graph['weight'] = df_graph['weight'].apply(np.float)

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


def copy_node_labels(label_path, output_dir):
    df_labels = pd.read_csv(label_path, sep=' ')
    df_labels['node'] = df_labels['node'].apply(lambda x: 'U' + str(x))
    for i in range(10):
        output_file_path = os.path.join(output_dir, str(i) + '.csv')
        df_labels.to_csv(output_file_path, sep='\t', index=False)


def build_graphs():
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


def copy_labels():
    input_dir = '/data/america_air/nodes_set'
    label_file = 'labels.csv'
    label_path = os.path.join(input_dir, label_file)
    output_dir = '/data/america_air/nodes_label'
    check_and_make_path(output_dir)
    copy_node_labels(label_path, output_dir)

    input_dir = '/data/europe_air/nodes_set'
    label_file = 'labels.csv'
    label_path = os.path.join(input_dir, label_file)
    output_dir = '/data/europe_air/nodes_label'
    check_and_make_path(output_dir)
    copy_node_labels(label_path, output_dir)


if __name__ == '__main__':
    # build_graphs()
    dataset = 'facebook_node'
    base_dir = '/data/' + dataset
    get_graph_from_nodes(file_path=os.path.join(base_dir, '0.input', '2008-12.csv'), node_file=os.path.join(base_dir, 'nodes_set/nodes.csv'),
                         output_node_dir=os.path.join(base_dir, 'nodes'), output_edge_dir=os.path.join(base_dir, '1.format'))
    # dataset = 'facebook_edge'
    # base_dir = '/data/' + dataset
    # get_graph_from_edges(file_path=os.path.join(base_dir, '0.input', '2008-12.csv'), node_file=os.path.join(base_dir, 'nodes_set/nodes.csv'),
    #                      output_node_dir=os.path.join(base_dir, 'nodes'), output_edge_dir=os.path.join(base_dir, '1.format'))

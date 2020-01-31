import os
import networkx as nx
import pandas as pd
import traceback
from time import time


def check_and_make_path(to_make):
    if not os.path.exists(to_make):
        os.makedirs(to_make)


def read_edgelist_from_dataframe(filename, full_node_list):
    dataframe = pd.read_csv(filename, sep='\t')
    # dataframe['weight'] = 1.0
    graph = nx.from_pandas_edgelist(dataframe, "from_id", "to_id", edge_attr='weight',
                                    create_using=nx.Graph)
    graph.add_nodes_from(full_node_list)

    return graph


def separate(info='', sep='=', num=5):
    print()
    if len(info) == 0:
        print(sep * (2 * num))
    else:
        print(sep * num, info, sep * num)
    print()


def time_filter_with_dict_param(func, **kwargs):
    try:
        t1 = time()
        func(**kwargs)
        t2 = time()
        print(func.__name__, " spends ", t2 - t1, 'ms')
    except Exception as e:
        traceback.print_exc()


def time_filter_with_tuple_param(func, *args):
    t1 = time()
    func(*args)
    t2 = time()
    print(func.__name__, " spends ", t2 - t1, 'ms')

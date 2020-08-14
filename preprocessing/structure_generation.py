# coding: utf-8
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import os
import multiprocessing
from utils import check_and_make_path, get_nx_graph, get_format_str


class StructureInfoGenerator:
    base_path: str
    origin_base_path: str
    core_base_path: str
    full_node_list: list
    node_num: int

    def __init__(self, base_path, origin_folder, core_folder, node_file):
        self.base_path = base_path
        self.origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
        self.core_base_path = os.path.abspath(os.path.join(base_path, core_folder))

        node_path = os.path.abspath(os.path.join(base_path, node_file))
        nodes_set = pd.read_csv(node_path, names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.node_num = len(self.full_node_list)

        check_and_make_path(self.core_base_path)

    # Most real-world graphs' k-core numbers start from 0. However, SBM graphs' k-core numbers start from 40(or 70), not start from 0.
    # Moreover, most real world graphs' k-core numbers are 0, 1, 2, 3, ... without any interval, while synthetic graph such as SBM's k-core numbers are 46,48,51,54, ..
    def get_kcore_graph(self, input_file, output_dir, sep='\t', core_list=None, degree_list=None):
        input_path = os.path.join(self.origin_base_path, input_file)
        graph = get_nx_graph(input_path, self.full_node_list, sep=sep)
        core_num_dict = nx.core_number(graph)
        print("unique core nums: ", len(np.unique(np.array(list(core_num_dict.values())))))
        max_core_num = max(list(core_num_dict.values()))
        print('file name: ', input_file, 'max core num: ', max_core_num)

        # x = list(graph.degree())
        # max_degree = max(list(zip(*x))[1])
        # print('max degree: ', max_degree)
        # core_list.append(max_core_num)
        # degree_list.append(max_degree)
        check_and_make_path(output_dir)

        format_str = get_format_str(max_core_num)
        for i in range(1, max_core_num + 1):
            k_core_graph = nx.k_core(graph, k=i, core_number=core_num_dict)
            k_core_graph.add_nodes_from(self.full_node_list)
            ###############################
            # This node_list is quit important, or it will change the graph adjacent matrix and cause bugs!!!
            A = nx.to_scipy_sparse_matrix(k_core_graph, nodelist=self.full_node_list)
            ###############################
            signature = format_str.format(i)
            sp.save_npz(os.path.join(output_dir, signature + '.npz'), A)

    def get_kcore_graph_all_time(self, sep='\t', worker=-1):
        print("getting k-core sub-graphs for all timestamps...")

        f_list = os.listdir(self.origin_base_path)
        f_list = sorted(f_list)

        length = len(f_list)
        if worker <= 0:
            core_list, degree_list = [], []
            for i, f_name in enumerate(f_list):
                self.get_kcore_graph(input_file=f_name, output_dir=os.path.join(self.core_base_path, f_name.split('.')[0]), sep=sep,
                                     core_list=core_list, degree_list=degree_list)
            # print('max max core: ', max(core_list))
            # print('max max degree: ', max(degree_list))
        else:
            worker = min(worker, length, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("\tstart " + str(worker) + " worker(s)")

            for i, f_name in enumerate(f_list):
                pool.apply_async(self.get_kcore_graph, (f_name, os.path.join(self.core_base_path, f_name.split('.')[0]), sep))
            pool.close()
            pool.join()
        print("got it...")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import os, multiprocessing, random
import sys, time
sys.path.append("..")
from RWTGCN.utils import check_and_make_path, get_sp_adj_mat, get_nx_graph, get_format_str

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

    def get_kcore_graph(self, input_file, output_dir):
        graph = get_nx_graph(input_file, self.full_node_list, sep='\t')
        core_num_dict = nx.core_number(graph)
        max_core_num = max(list(core_num_dict.values()))
        print('max core num: ', max_core_num)
        check_and_make_path(output_dir)

        format_str = get_format_str(max_core_num)
        for i in range(1, max_core_num + 1):
            k_core_graph = nx.k_core(graph, k=i, core_number=core_num_dict)
            k_core_graph.add_nodes_from(np.arange(self.node_num))
            A = nx.to_scipy_sparse_matrix(k_core_graph)
            signature = format_str.format(i)
            sp.save_npz(os.path.join(output_dir, signature + ".npz"), A)
        return

    def get_kcore_graph_all_time(self, worker=-1):
        print("getting k-core graph for all timestamps...")

        f_list = os.listdir(self.origin_base_path)
        length = len(f_list)
        if worker <= 0:
            for i, f_name in enumerate(f_list):
                self.get_kcore_graph(
                    input_file=os.path.join(self.origin_base_path, f_name),
                    output_dir=os.path.join(self.core_base_path, f_name.split('.')[0]), )
        else:
            worker = min(worker, length, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("\tstart " + str(worker) + " worker(s)")

            for i, f_name in enumerate(f_list):
                pool.apply_async(self.get_kcore_graph, (
                    os.path.join(self.origin_base_path, f_name),
                    os.path.join(self.core_base_path, f_name.split('.')[0]),))
            pool.close()
            pool.join()
        print("got it...")

if __name__ == "__main__":
    s = StructureInfoGenerator(base_path="..\\data\\email-eu", input_folder="1.format",
                                   output_folder="RWT-GCN\\structural_network", node_file="nodes_set\\nodes.csv",
                                   hop=1)
    s.get_structural_info_all_time(worker=-1)
    s.get_kcore_graph_all_time(worker=-1)

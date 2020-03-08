import numpy as np
import pandas as pd
import os, multiprocessing, time
import networkx as nx
from numpy import random
import scipy.sparse as sp
import sys
sys.path.append("..")
from RWTGCN.utils import check_and_make_path, get_sp_adj_mat

class WalkGenerator:
    base_path: str
    origin_base_path: str
    walk_pair_base_path: str
    node_freq_base_path: str
    full_node_list: list
    walk_time: int
    walk_length: int

    def __init__(self, base_path, origin_folder, walk_pair_folder, node_freq_folder,  node_file, walk_time=100, walk_length=5):
        self.base_path = base_path
        self.origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder))
        self.walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        self.node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))

        node_path = os.path.abspath(os.path.join(base_path, node_file))
        nodes_set = pd.read_csv(node_path, names=['node'])
        self.full_node_list = nodes_set['node'].tolist()

        self.walk_time = walk_time
        self.walk_length = walk_length

        check_and_make_path(self.walk_pair_base_path)
        check_and_make_path(self.node_freq_base_path)

    def get_walk_info(self, f_name, original_graph_path, weight=True):
        import RWTGCN.preprocessing.helper as helper
        print('f_name = ', f_name)
        t1 = time.time()
        # only random walk on original graph
        spadj = get_sp_adj_mat(original_graph_path, self.full_node_list)
        helper.random_walk(spadj, self.walk_pair_base_path, self.node_freq_base_path, f_name,
                           self.walk_length, self.walk_time, weight=weight)
        t2 = time.time()
        print('random walk tot time', t2 - t1, ' seconds!')

    def get_walk_info_all_time(self, worker=-1):
        print("all file(s) in folder transform to tensor...")
        f_list = os.listdir(self.origin_base_path)

        if worker <= 0:
            for i, f_name in enumerate(f_list):
                original_graph_path = os.path.join(self.origin_base_path, f_name)
                # t1 = time.time()
                self.get_walk_info(f_name, original_graph_path=original_graph_path)
                #t2 = time.time()
                #print('generate tensor time: ', t2 - t1, ' seconds!')
        else:
            worker = min(os.cpu_count(), worker)
            pool = multiprocessing.Pool(processes=worker)
            print("\t\tstart " + str(worker) + " worker(s)")
            for i, f_name in enumerate(f_list):
                original_graph_path = os.path.join(self.origin_base_path, f_name)
                pool.apply_async(self.get_walk_info, (f_name, original_graph_path,))
            pool.close()
            pool.join()

if __name__ == "__main__":
    tg = WalkGenerator(base_path="..\\data\\email-eu", input_folder="1.format",
                         output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv",
                         walk_time=100, walk_length=5)
    tg.get_walk_info_all_time(worker=-1)
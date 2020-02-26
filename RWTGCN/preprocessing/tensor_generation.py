import numpy as np
import pandas as pd
import os, multiprocessing, time
import networkx as nx
from numpy import random
import scipy.sparse as sp
import sys
sys.path.append("..")
from RWTGCN.utils import check_and_make_path, build_graph

class TensorGenerator:
    base_path: str
    origin_base_path: str
    structure_base_path: str
    walk_pair_base_path: str
    node_freq_base_path: str
    walk_tensor_base_path: str
    full_node_list: list
    walk_time: int
    walk_length: int
    prob: float

    def __init__(self, base_path, origin_folder, structure_folder, walk_pair_folder, node_freq_folder,  walk_tensor_folder, node_file,
                 walk_time=100, walk_length=5, prob=0.5):
        self.base_path = base_path
        self.origin_base_path = '' if origin_folder == '' else os.path.abspath(os.path.join(base_path, origin_folder))
        #print('origin base: ', self.origin_base_path)
        self.structure_base_path = '' if structure_folder == '' else os.path.abspath(os.path.join(base_path, structure_folder))
        #print('structure base: ', self.structure_base_path)
        self.walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
        #print('walk pair:', self.walk_pair_base_path)
        self.node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))
        #print('node freq: ', self.node_freq_base_path)
        self.walk_tensor_base_path = '' if walk_tensor_folder == '' else os.path.abspath(os.path.join(base_path, walk_tensor_folder))
        #print('walk tensor: ', self.walk_tensor_base_path)
        node_path = os.path.abspath(os.path.join(base_path, node_file))
        #print('node file: ', node_path)
        nodes_set = pd.read_csv(node_path, names=['node'])
        self.full_node_list = nodes_set['node'].tolist()

        self.walk_time = walk_time
        self.walk_length = walk_length
        self.prob = prob

        check_and_make_path(self.walk_pair_base_path)
        check_and_make_path(self.node_freq_base_path)
        check_and_make_path(self.walk_tensor_base_path)

    def generate_tensor(self, f_name, original_graph_path, structural_graph_path, weight=True):
        import RWTGCN.preprocessing.helper as helper
        print('f_name = ', f_name)

        f_folder = f_name.split('.')[0]
        tensor_dir_path = '' if self.walk_tensor_base_path == '' else os.path.abspath(os.path.join(self.walk_tensor_base_path, f_folder))
        eps = 1e-8
        t1 = time.time()
        # only random walk on original graph
        if np.abs(self.prob - 1) < eps:
            assert original_graph_path != ''
            if tensor_dir_path != '':
                check_and_make_path(tensor_dir_path)
            original_graph = build_graph(original_graph_path, self.full_node_list)
            helper.random_walk(original_graph, self.walk_pair_base_path, self.node_freq_base_path, f_name,
                               tensor_dir_path, self.walk_length, self.walk_time, weight=weight, tensor_flag=(tensor_dir_path != ''))
        # only random walk on structural graph
        elif abs(self.prob) < eps:
            assert structural_graph_path != ''
            if tensor_dir_path != '':
                check_and_make_path(tensor_dir_path)
            structural_graph = build_graph(structural_graph_path, self.full_node_list)
            helper.random_walk(structural_graph, self.walk_pair_base_path, self.node_freq_base_path, f_name,
                               tensor_dir_path, self.walk_length, self.walk_time, weight=weight, tensor_flag=(tensor_dir_path != ''))
        # hybrid random walk on original graph and structural graph
        else:
            assert (original_graph_path != '' and structural_graph_path != '' and tensor_dir_path != '')
            check_and_make_path(tensor_dir_path)
            original_graph = build_graph(original_graph_path, self.full_node_list)
            structural_graph = build_graph(structural_graph_path, self.full_node_list)
            helper.hybrid_random_walk(original_graph, structural_graph, self.walk_pair_base_path, self.node_freq_base_path,
                                      f_name, tensor_dir_path, self.walk_length, self.walk_time, self.prob, weight)
        t2 = time.time()
        print('random walk tot time', t2 - t1, ' seconds!')

    def generate_tensor_all_time(self, worker=-1):
        print("all file(s) in folder transform to tensor...")
        f_list = os.listdir(self.origin_base_path)

        if worker <= 0:
            for i, f_name in enumerate(f_list):
                original_graph_path = os.path.join(self.origin_base_path, f_name)
                structural_graph_path = os.path.join(self.structure_base_path, f_name)
                # t1 = time.time()
                self.generate_tensor(f_name, original_graph_path=original_graph_path,
                                     structural_graph_path=structural_graph_path)
                #t2 = time.time()
                #print('generate tensor time: ', t2 - t1, ' seconds!')
        else:
            worker = min(os.cpu_count(), worker)
            pool = multiprocessing.Pool(processes=worker)
            print("\t\tstart " + str(worker) + " worker(s)")
            for i, f_name in enumerate(f_list):
                original_graph_path = os.path.join(self.origin_base_path, f_name)
                structural_graph_path =  os.path.join(self.structure_base_path, f_name)
                pool.apply_async(self.generate_tensor, (f_name, original_graph_path, structural_graph_path))
            pool.close()
            pool.join()

if __name__ == "__main__":
    tg = TensorGenerator(base_path="..\\data\\email-eu", input_folder="1.format",
                         output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv",
                         walk_time=100, walk_length=5, prob=0.5)
    tg.generate_tensor_all_time(worker=-1)
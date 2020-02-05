import numpy as np
import pandas as pd
import os, multiprocessing, time
from numpy import random
import scipy.sparse as sp
from RWTGCN.utils import check_and_make_path, read_edgelist_from_dataframe

class TensorGenerator:
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list
    walk_time: int
    walk_length: int
    prob: float

    # 这里的孤点(在nodelist而不在edgelist中的一定不会有游走序列，所以就不再图里添加这些孤点了)
    def __init__(self, base_path, input_folder, output_folder, node_file, walk_time=100, walk_length=5, prob=0.5):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()

        self.walk_time = walk_time
        self.walk_length = walk_length
        self.prob = prob

        check_and_make_path(self.output_base_path)
        tem_dir = ['walk_pairs', 'node_freq', 'walk_tensor']
        for tem in tem_dir:
            check_and_make_path(os.path.join(self.output_base_path, tem))

    def generate_tensor(self, f_name, original_graph_path, structural_graph_path, weight=True):
        print('f_name = ', f_name)
        f_folder = f_name.split('.')[0]
        walk_dir_path = os.path.join(self.output_base_path, 'walk_pairs')
        freq_dir_path = os.path.join(self.output_base_path, 'node_freq')
        tensor_dir_path = os.path.join(self.output_base_path, 'walk_tensor', f_folder)
        check_and_make_path(tensor_dir_path)

        original_graph = read_edgelist_from_dataframe(original_graph_path, self.full_node_list)
        structural_graph = read_edgelist_from_dataframe(structural_graph_path, self.full_node_list)

        # try:
        #     import RWTGCN.preprocessing.helper as helper
        #     return helper.random_walk(original_graph, structural_graph, self.full_node_list,
        #                               walk_dir_path, freq_dir_path, f_name, tensor_dir_path,
        #                               self.walk_length, self.walk_time, self.prob, weight)
        # except:
        #     pass
        import RWTGCN.utils as utils
        return utils.random_walk(original_graph, structural_graph, self.full_node_list,
                                 walk_dir_path, freq_dir_path, f_name, tensor_dir_path,
                                 self.walk_length, self.walk_time, self.prob, weight)

    def generate_tensor_all_time(self, worker=-1):
        print("all file(s) in folder transform to tensor...")
        f_list = os.listdir(self.input_base_path)

        if worker <= 0:
            for i, f_name in enumerate(f_list):
                original_graph_path = os.path.join(self.input_base_path, f_name)
                structural_graph_path = os.path.join(self.output_base_path, 'structural_network', f_name)
                t1 = time.time()
                self.generate_tensor(f_name, original_graph_path=original_graph_path,
                                     structural_graph_path=structural_graph_path)
                t2 = time.time()
                print('generate tensor time: ', t2 - t1, ' seconds!')
        else:
            worker = min(os.cpu_count(), self.walk_length, worker)
            pool = multiprocessing.Pool(processes=worker)
            print("\t\tstart " + str(worker) + " worker(s)")
            for i, f_name in enumerate(f_list):
                original_graph_path = os.path.join(self.input_base_path, f_name)
                structural_graph_path = os.path.join(self.output_base_path, 'structural_network', f_name)
                pool.apply_async(self.generate_tensor, (f_name, original_graph_path, structural_graph_path))
            pool.close()
            pool.join()


if __name__ == "__main__":
    tg = TensorGenerator(base_path="..\\data\\email-eu", input_folder="1.format",
                         output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv",
                         walk_time=100, walk_length=5, prob=0.5)
    tg.generate_tensor_all_time(worker=-1)
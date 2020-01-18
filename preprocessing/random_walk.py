#coding: utf-8
import numpy as np
import pandas as pd
import os, random, multiprocessing, json
from utils import dir_helper
from utils.read_format_data import read_edgelist_from_dataframe

class HybridRandomWalk:
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list
    walk_time: int
    walk_length: int
    p: float

    def __init__(self, base_path, input_folder, output_folder, node_file, walk_time=100, walk_length=10, p=0.5):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.walk_time = walk_time
        self.walk_length = walk_length
        self.p = p

        dir_helper.check_and_make_path(self.output_base_path)
        tem_dir = ['walk_sequences']
        for tem in tem_dir:
            dir_helper.check_and_make_path(os.path.join(self.output_base_path, tem))
        return

    def random_with_weight(self, neighbors: list, weight: list):
        weight = np.array(weight, dtype=float)
        # sum = weight.sum()
        # weight /= sum
        ran = random.random()
        clen = len(neighbors)
        for i in range(clen):
            ran -= weight[i]
            if ran <= 0:
                return neighbors[i]

    def random_walk_from_node(self, original_graph, structural_graph, start_node):
        eps = 1e-8
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            rd = random.random()
            if rd <= self.p + eps:
                cur_nbrs = list(original_graph.neighbors(cur))
            else:
                cur_nbrs = list(structural_graph.neighbors(cur))
            # print('cur: ', cur_nbrs, ', type: ', type(cur_nbrs))
            if len(cur_nbrs) == 0:
                break
            walk.append(random.choice(cur_nbrs))
        return walk

    def random_walk_from_node_with_weight(self, original_graph, structural_graph, start_node):
        eps = 1e-8
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            rd = random.random()
            if rd <= self.p + eps:  # choose origin network
                candidates = list(original_graph.neighbors(cur))
                candidates_weight = [original_graph[cur][i]['weight'] for i in candidates]
            else:  # choose structural network
                candidates = list(structural_graph.neighbors(cur))
                candidates_weight = [structural_graph[cur][i]['weight'] for i in candidates]
            walk.append(self.random_with_weight(neighbors=candidates, weight=candidates_weight))
        return walk

    def get_walk_sequences(self, original_graph_path, structural_graph_path, output_file=None, file_num=None, i=None):
        # print('\t', str(file_num - i), ' file(s) left')
        # if os.path.exists(output_file):
        #     print('\t', output_file, "exist")
        #     print('\t', str(file_num - i), ' finished')
        #     return

        original_graph = read_edgelist_from_dataframe(original_graph_path, self.full_node_list)
        structural_graph = read_edgelist_from_dataframe(structural_graph_path, self.full_node_list)
        walks = []
        nodes = self.full_node_list.copy()
        for iter in range(self.walk_time):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk_from_node(original_graph=original_graph,
                                                        structural_graph=structural_graph,
                                                        start_node=node))
        # with open(output_file, 'w') as fp:
        #     json.dump(walks, fp)
        return walks

    def get_walk_sequences_with_weight(self, original_graph_path, structural_graph_path, output_file=None, file_num=None, i=None):
        # print('\t', str(file_num - i), ' file(s) left')
        # if os.path.exists(output_file):
        #     print('\t', output_file, "exist")
        #     print('\t', str(file_num - i), ' finished')
        #     return

        original_graph = read_edgelist_from_dataframe(original_graph_path, self.full_node_list)
        structural_graph = read_edgelist_from_dataframe(structural_graph_path, self.full_node_list)
        walks = []
        nodes = self.full_node_list.copy()
        for iter in range(self.walk_time):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk_from_node_with_weight(original_graph=original_graph,
                                                                    structural_graph=structural_graph,
                                                                    start_node=node))
        # with open(output_file, 'w') as fp:
        #     json.dump(walks, fp)
        return walks

    def get_walk_sequences_all_time(self, worker=-1):
        print("getting all timestamps random walk sequences...")

        f_list = os.listdir(self.input_base_path)
        length = len(f_list)

        if worker <= 0:
            for i, f_name in enumerate(f_list):
                self.get_walk_sequences(original_graph_path=os.path.join(self.input_base_path, f_name),
                                        structural_graph_path=os.path.join(self.output_base_path, 'structural_network', f_name),
                                        #output_file=os.path.join(self.output_base_path, 'walk_sequences', f_name),
                                        file_num=length, i=i)

        else:
            worker = min(worker, length, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("\tstart " + str(worker) + " worker(s)")

            for i, f_name in enumerate(f_list):
                pool.apply_async(self.get_walk_sequences, (os.path.join(self.input_base_path, f_name),
                                     os.path.join(self.output_base_path, 'structural_network',f_name),
                                     # os.path.join(self.output_base_path, 'walk_sequences', f_name),
                                     length, i))

            pool.close()
            pool.join()
        print("got it...")
        return

    def get_walk_sequences_with_weight_all_time(self, worker=-1):
        print("getting all timestamps weighted random walk sequences...")

        f_list = os.listdir(self.input_base_path)
        length = len(f_list)

        if worker <= 0:
            for i, f_name in enumerate(f_list):
                self.get_walk_sequences_with_weight(original_graph_path=os.path.join(self.input_base_path, f_name),
                                        structural_graph_path=os.path.join(self.output_base_path, 'structural_network', f_name),
                                        output_file=os.path.join(self.output_base_path, 'walk_sequences', f_name),
                                        file_num=length, i=i)

        else:
            worker = min(worker, length, os.cpu_count())
            pool = multiprocessing.Pool(processes=worker)
            print("\tstart " + str(worker) + " worker(s)")

            for i, f_name in enumerate(f_list):
                pool.apply_async(self.get_walk_sequences_with_weight, (os.path.join(self.input_base_path, f_name),
                                     os.path.join(self.output_base_path, 'structural_network',f_name),
                                     os.path.join(self.output_base_path, 'walk_sequences', f_name),
                                     length, i))
            pool.close()
            pool.join()
        print("got it...")
        return

if __name__ == '__main__':
    random_walk = HybridRandomWalk(base_path="..\\data\\email-eu", input_folder="1.format",
                                   output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv",
                                   walk_time=100, walk_length=10, p=0.5)
    random_walk.get_walk_sequences_all_time(worker=-1)

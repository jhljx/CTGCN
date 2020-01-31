import numpy as np
import pandas as pd
import os, time
from numpy import random
from RWTGCN.utils import check_and_make_path, read_edgelist_from_dataframe


class HybridRandomWalk:
    base_path: str
    input_base_path: str
    output_base_path: str
    full_node_list: list
    walk_time: int
    walk_length: int
    p: float

    def __init__(self, base_path, input_folder, output_folder, node_file, walk_time=100, walk_length=5, p=0.5):
        self.base_path = base_path
        self.input_base_path = os.path.join(base_path, input_folder)
        self.output_base_path = os.path.join(base_path, output_folder)

        nodes_set = pd.read_csv(os.path.join(base_path, node_file), names=['node'])
        self.full_node_list = nodes_set['node'].tolist()
        self.walk_time = walk_time
        self.walk_length = walk_length
        self.p = p

        check_and_make_path(self.output_base_path)
        return

    def random_walk_from_node(self, original_graph_dict, structural_graph_dict, start_node):
        eps = 1e-8
        walk = [start_node]
        while len(walk) < self.walk_length + 1:
            cur = walk[-1]
            rd = random.random()
            if rd <= self.p + eps:
                cur_nbrs = original_graph_dict[cur]
            else:
                cur_nbrs = structural_graph_dict[cur]
            if len(cur_nbrs) == 0:
                break
            walk.append(random.choice(cur_nbrs))
        return walk

    def random_walk_from_node_with_weight(self, original_graph_dict, structural_graph_dict, start_node):
        eps = 1e-8
        walk = [start_node]
        while len(walk) < self.walk_length + 1:
            cur = walk[-1]
            rd = random.random()
            if rd <= self.p + eps:  # choose origin network
                candidates = original_graph_dict[cur]['neighbor']
                candidates_weight = original_graph_dict[cur]['weight']
            else:  # choose structural network
                candidates = structural_graph_dict[cur]['neighbor']
                candidates_weight = structural_graph_dict[cur]['weight']
            if len(candidates) == 0:
                break
            walk.append(random.choice(candidates, p=candidates_weight))
        return walk

    def get_walk_sequences(self, original_graph_path, structural_graph_path):
        original_graph = read_edgelist_from_dataframe(original_graph_path, self.full_node_list)
        structural_graph = read_edgelist_from_dataframe(structural_graph_path, self.full_node_list)
        walks = []
        nodes = self.full_node_list.copy()
        original_graph_dict, structural_graph_dict = {}, {}
        # preprocessing
        for node in nodes:
            original_graph_dict[node] = list(original_graph.neighbors(node))
            structural_graph_dict[node] = list(structural_graph.neighbors(node))
        # random walk
        for iter in range(self.walk_time):
            # random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk_from_node(original_graph_dict=original_graph_dict,
                                                        structural_graph_dict=structural_graph_dict,
                                                        start_node=node))
        return walks

    def get_walk_sequences_with_weight(self, original_graph_path, structural_graph_path):
        original_graph = read_edgelist_from_dataframe(original_graph_path, self.full_node_list)
        structural_graph = read_edgelist_from_dataframe(structural_graph_path, self.full_node_list)
        walks = []
        nodes = self.full_node_list.copy()

        original_graph_dict, structural_graph_dict = {}, {}
        # preprocessing
        for node in nodes:
            original_neighbors = list(original_graph.neighbors(node))
            original_weight = np.array([original_graph[node][i]['weight'] for i in original_neighbors])
            original_graph_dict[node] = {'neighbor': original_neighbors}
            original_graph_dict[node]['weight'] = original_weight / original_weight.sum()

            structural_neighbors = list(structural_graph.neighbors(node))
            structural_weight = np.array([structural_graph[node][i]['weight'] for i in structural_neighbors])
            structural_graph_dict[node] = {'neighbor': structural_neighbors}
            structural_graph_dict[node]['weight'] = structural_weight / structural_weight.sum()
        # random walk
        for iter in range(self.walk_time):
            # random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk_from_node_with_weight(original_graph_dict=original_graph_dict,
                                                                    structural_graph_dict=structural_graph_dict,
                                                                    start_node=node))
        return walks


if __name__ == '__main__':
    start = time.time()
    random_walk = HybridRandomWalk(base_path="..\\data\\email-eu", input_folder="1.format",
                                   output_folder="RWT-GCN", node_file="nodes_set\\nodes.csv",
                                   walk_time=100, walk_length=5, p=0.5)
    walks = random_walk.get_walk_sequences_with_weight(
        original_graph_path=os.path.join(random_walk.input_base_path, '1970_01.csv'),
        structural_graph_path=os.path.join(random_walk.output_base_path, 'structural_network',
                                           '1970_01.csv'))
    print('sequence num: ', len(walks))
    end = time.time()
    print('total time cost: ', (end - start), ' seconds.')

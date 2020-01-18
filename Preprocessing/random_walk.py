import numpy as np
import pandas as pd
import os, random
import networkx as nx
from utils.read_format_data import read_edgelist_from_dataframe

class RandomWalk:
    graph: nx.Graph
    node_list: list
    walk_time: int
    walk_length: int

    def __init__(self, graph_path, node_path, walk_time=100, walk_length=10):
        df_nodes = pd.read_csv(node_path, names=['node'])
        self.node_list = df_nodes['node'].tolist()
        self.graph = read_edgelist_from_dataframe(graph_path, self.node_list)
        self.walk_time = walk_time
        self.walk_length = walk_length
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

    def random_walk_from_node(self, start_node):
        eps = 1e-8
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.graph.neighbors(cur))
            if len(cur_nbrs) == 0:
                break
            walk.append(random.choice(cur_nbrs))
        return walk

    def random_walk_from_node_with_weight(self, start_node):
        eps = 1e-8
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            candidates = list(self.graph.neighbors(cur))
            candidates_weight = [self.graph[cur][i]['weight'] for i in candidates]
            walk.append(self.random_with_weight(neighbors=candidates, weight=candidates_weight))
        return walk

    def get_walk_sequence(self):
        walks = []
        nodes = self.node_list.copy()
        for iter in range(self.walk_time):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk_from_node(start_node=node))
        return walks

    def get_walk_sequence_with_weight(self):
        walks = []
        nodes = self.node_list.copy()
        for iter in range(self.walk_time):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.random_walk_from_node_with_weight(start_node=node))
        return walks

class HybridRandomWalk(RandomWalk):
    structural_graph: nx.Graph
    p: float

    def __init__(self, origin_graph_path, structural_graph_path, node_path, walk_time=100, walk_length=10, p=0.5):
        super(HybridRandomWalk, self).__init__(origin_graph_path, node_path, walk_time, walk_length)
        self.structural_graph = read_edgelist_from_dataframe(structural_graph_path, self.node_list)
        self.p = p
        return

    def random_walk_from_node(self, start_node):
        eps = 1e-8
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            rd = random.random()
            if rd <= self.p + eps:
                cur_nbrs = list(self.graph.neighbors(cur))
            else:
                cur_nbrs = list(self.structural_graph.neighbors(cur))
            # print('cur: ', cur_nbrs, ', type: ', type(cur_nbrs))
            if len(cur_nbrs) == 0:
                break
            walk.append(random.choice(cur_nbrs))
        return walk

    def random_walk_from_node_with_weight(self, start_node):
        eps = 1e-8
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            rd = random.random()
            if rd <= self.p + eps:  # choose origin network
                candidates = list(self.graph.neighbors(cur))
                candidates_weight = [self.graph[cur][i]['weight'] for i in candidates]
            else:  # choose structural network
                candidates = list(self.structural_graph.neighbors(cur))
                candidates_weight = [self.structural_graph[cur][i]['weight'] for i in candidates]
            walk.append(self.random_with_weight(neighbors=candidates, weight=candidates_weight))
        return walk


if __name__ == '__main__':
    base_path = "..\\data\\email-eu"
    origin_graph_path = os.path.join(base_path, '1.format', '1970_01.csv')
    structural_graph_path = os.path.join(base_path, 'RWT-GCN', 'structural_network', '1970_01.csv')
    node_path = os.path.join(base_path, 'nodes_set', 'nodes.csv')
    random_walk = HybridRandomWalk(origin_graph_path, structural_graph_path, node_path, walk_time=100, walk_length=10, p=0.5)
    walks = random_walk.get_walk_sequence()
